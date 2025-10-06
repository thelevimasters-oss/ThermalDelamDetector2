# processing.py
"""
Core image processing logic for Thermal Delamination Detector.

This module provides:
- Loading and temperature-map extraction from supported images
- Normalization, optional Gaussian blur
- Hotspot thresholding by percentile
- Morphological opening/closing using Pillow's Min/Max filters
- Small-cluster removal via connected-component filtering
- Colormap rendering and overlay generation
- Statistics computation (overall and for ROIs)
- Optional simple GPU acceleration if PyTorch is available (float ops only)

Dependencies: Pillow, NumPy (required). PyTorch (optional).
Python 3.10+
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Optional GPU with PyTorch
try:
    import torch  # type: ignore

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


SUPPORTED_EXTS = {".rjpg", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass
class ProcessingParams:
    hotspot_percentile: float = 97.0  # 50..100
    min_cluster_size: int = 45
    opening_iterations: int = 1
    closing_iterations: int = 1
    kernel_size: int = 3  # must be odd
    overlay_opacity: float = 0.6  # 0..1
    gaussian_sigma: float = 0.0  # pre-blur; 0 = disabled
    colormap: str = "blue_red"  # options: blue_red, jet, viridis
    use_gpu: bool = False  # if PyTorch available

    def clamped(self) -> "ProcessingParams":
        k = self.kernel_size if self.kernel_size % 2 == 1 else max(1, self.kernel_size - 1)
        return ProcessingParams(
            hotspot_percentile=float(np.clip(self.hotspot_percentile, 50.0, 100.0)),
            min_cluster_size=max(0, int(self.min_cluster_size)),
            opening_iterations=max(0, int(self.opening_iterations)),
            closing_iterations=max(0, int(self.closing_iterations)),
            kernel_size=k,
            overlay_opacity=float(np.clip(self.overlay_opacity, 0.0, 1.0)),
            gaussian_sigma=max(0.0, float(self.gaussian_sigma)),
            colormap=self.colormap,
            use_gpu=bool(self.use_gpu and TORCH_AVAILABLE),
        )


@dataclass
class ProcessingOutputs:
    temperature_norm: np.ndarray  # float32, [0,1], HxW
    hotspot_mask: np.ndarray  # bool, HxW
    overlay_rgb: Image.Image  # PIL RGB image
    stats: Dict[str, Union[int, float, Dict[str, float]]]


def load_image(path: str) -> Image.Image:
    """
    Load an image using Pillow, converting to RGB.
    Supports .rjpg via Pillow if decodable as JPEG; otherwise will raise.
    """
    img = Image.open(path)
    img.load()
    if img.mode not in ("RGB", "L", "I;16", "I", "F"):
        img = img.convert("RGB")
    return img


def _image_to_temperature_gray(img: Image.Image, use_gpu: bool = False) -> np.ndarray:
    """
    Convert image to a float32 grayscale 'temperature-like' array.
    If image has a single channel float/int mode, normalize its values.
    Otherwise, convert to luminance.
    """
    # Convert to grayscale float
    if img.mode in ("F",):
        arr = np.array(img, dtype=np.float32)
    elif img.mode in ("I;16", "I"):
        arr = np.array(img, dtype=np.float32)
    else:
        gray = ImageOps.grayscale(img)
        arr = np.array(gray, dtype=np.float32)

    # Handle non-finite values
    arr = np.nan_to_num(arr, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)

    # Normalize to [0,1] robustly (ignore top/bottom 0.5% to reduce outlier effect)
    lo = np.percentile(arr, 0.5)
    hi = np.percentile(arr, 99.5)
    if hi <= lo:
        hi = lo + 1.0
    norm = (arr - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)

    if use_gpu and TORCH_AVAILABLE:
        # Move to GPU then back (demonstrates GPU pathway / consistency)
        t = torch.from_numpy(norm).to("cuda" if torch.cuda.is_available() else "cpu")
        norm = t.clamp(0, 1).float().cpu().numpy()

    return norm


def _gaussian_blur_numpy(arr: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur using Pillow's GaussianBlur via Image interface for simplicity.
    If sigma <= 0: return arr unchanged.
    """
    if sigma <= 0.0:
        return arr
    # Convert to 8-bit for filter speed, then back to float
    arr8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(arr8, mode="L").filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    out = np.array(img, dtype=np.float32) / 255.0
    return out


def _percentile_threshold(arr01: np.ndarray, percentile: float) -> float:
    p = float(np.clip(percentile, 0.0, 100.0))
    return float(np.percentile(arr01, p))


def _binary_open_close(mask: np.ndarray, kernel_size: int, open_iter: int, close_iter: int) -> np.ndarray:
    """
    Morphology using Pillow Min/Max filters on a binary image.
    Opening = erosion (MinFilter) then dilation (MaxFilter); Closing = dilation then erosion.
    """
    if kernel_size < 1:
        return mask
    k = kernel_size if kernel_size % 2 == 1 else kernel_size - 1
    img = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    for _ in range(max(0, int(open_iter))):
        img = img.filter(ImageFilter.MinFilter(size=k))
        img = img.filter(ImageFilter.MaxFilter(size=k))
    for _ in range(max(0, int(close_iter))):
        img = img.filter(ImageFilter.MaxFilter(size=k))
        img = img.filter(ImageFilter.MinFilter(size=k))
    return (np.array(img) > 0).astype(bool)


def _remove_small_clusters(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Remove connected components smaller than min_size (8-connectivity) using an efficient flood fill.
    """
    if min_size <= 1:
        return mask
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out = mask.copy()
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    idxs = np.argwhere(mask)
    for y, x in idxs:
        if not mask[y, x] or visited[y, x]:
            continue
        # BFS to collect component
        stack = [(y, x)]
        comp = []
        visited[y, x] = True
        while stack:
            cy, cx = stack.pop()
            comp.append((cy, cx))
            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(comp) < min_size:
            for (cy, cx) in comp:
                out[cy, cx] = False
    return out


def _colormap(name: str, arr01: np.ndarray) -> np.ndarray:
    """
    Map [0,1] grayscale to RGB uint8 via simple LUTs for 'blue_red', 'jet', 'viridis' (approx).
    """
    name = (name or "blue_red").lower()
    x = np.clip(arr01, 0.0, 1.0)
    if name == "jet":
        # Approximate 'jet'
        r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    elif name == "viridis":
        # Simple approximation using polynomials
        r = 0.2803 + 1.5 * x - 1.3 * x**2
        g = 0.063 + 2.2 * x - 2.1 * x**2
        b = 0.245 + 1.3 * x - 1.1 * x**2
        r = np.clip(r, 0, 1)
        g = np.clip(g, 0, 1)
        b = np.clip(b, 0, 1)
    else:
        # blue_red default: blue -> cyan -> yellow -> red
        r = np.clip(2 * x, 0, 1)
        g = np.clip(2 * np.minimum(x, 1 - x), 0, 1)
        b = np.clip(2 * (1 - x), 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    return rgb


def _blend_overlay(base_rgb: np.ndarray, colorized: np.ndarray, hotspot_mask: np.ndarray, opacity: float) -> np.ndarray:
    """
    Blend colorized base with red-highlighted hotspots at given opacity.
    Hotspots get forced to red color (255,0,0) before blending.
    """
    h, w, _ = colorized.shape
    red = np.zeros_like(colorized)
    red[..., 0] = 255
    cm = colorized.copy()
    cm[hotspot_mask] = red[hotspot_mask]
    alpha = float(np.clip(opacity, 0.0, 1.0))
    out = (alpha * cm + (1.0 - alpha) * base_rgb).astype(np.uint8)
    return out


def compute_stats(temperature01: np.ndarray, mask: np.ndarray) -> Dict[str, Union[int, float]]:
    """
    Compute simple statistics on normalized temp and hotspot area.
    """
    total_pixels = int(temperature01.size)
    hotspot_pixels = int(mask.sum())
    coverage = (hotspot_pixels / total_pixels) * 100.0 if total_pixels else 0.0
    t_all = temperature01.ravel()
    t_hot = temperature01[mask] if hotspot_pixels > 0 else np.array([], dtype=np.float32)

    def safe_stats(a: np.ndarray) -> Tuple[float, float, float]:
        if a.size == 0:
            return float("nan"), float("nan"), float("nan")
        return float(np.min(a)), float(np.mean(a)), float(np.max(a))

    mn, av, mx = safe_stats(t_all)
    hmn, hav, hmx = safe_stats(t_hot)
    return {
        "total_pixels": total_pixels,
        "hotspot_pixels": hotspot_pixels,
        "hotspot_coverage_percent": coverage,
        "temperature_min": mn,
        "temperature_avg": av,
        "temperature_max": mx,
        "hotspot_temperature_min": hmn,
        "hotspot_temperature_avg": hav,
        "hotspot_temperature_max": hmx,
    }


def apply_roi(arr: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Crop array by ROI (x, y, w, h). If None, return original.
    """
    if roi is None:
        return arr
    x, y, w, h = roi
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)
    return arr[y:y2, x:x2]


def process_image(
    img: Image.Image,
    params: ProcessingParams,
    roi_xywh: Optional[Tuple[int, int, int, int]] = None,
) -> ProcessingOutputs:
    """
    Full processing pipeline.
    """
    p = params.clamped()

    # Convert to temperature map [0,1]
    temp01_full = _image_to_temperature_gray(img, use_gpu=p.use_gpu)
    # Optional ROI for analysis mask; but previews/overlays render on full image
    temp01_roi = apply_roi(temp01_full, roi_xywh)

    # Optional blur (applied to ROI temp for thresholding)
    temp_for_thresh = _gaussian_blur_numpy(temp01_roi, p.gaussian_sigma)

    # Threshold to hotspots by percentile on ROI region
    thr = _percentile_threshold(temp_for_thresh, p.hotspot_percentile)
    mask_roi = temp_for_thresh >= thr

    # Morphology (opening/closing) on ROI
    mask_roi = _binary_open_close(mask_roi, p.kernel_size, p.opening_iterations, p.closing_iterations)

    # Remove small clusters
    mask_roi = _remove_small_clusters(mask_roi, p.min_cluster_size)

    # Recompose mask to full size if ROI was used
    if roi_xywh is not None:
        full_mask = np.zeros_like(temp01_full, dtype=bool)
        x, y, w, h = roi_xywh
        full_mask[y : y + h, x : x + w] = mask_roi
        mask_full = full_mask
    else:
        mask_full = mask_roi

    # Colorize temperature map
    colorized = _colormap(p.colormap, temp01_full)
    base_rgb = np.array(img.convert("RGB"))
    overlay_rgb = _blend_overlay(base_rgb, colorized, mask_full, p.overlay_opacity)
    overlay_img = Image.fromarray(overlay_rgb, mode="RGB")

    # Stats computed on ROI region (if ROI set) otherwise full image
    stats_arr = temp01_roi if roi_xywh is not None else temp01_full
    stats_mask = mask_roi if roi_xywh is not None else mask_full
    stats = compute_stats(stats_arr, stats_mask)

    return ProcessingOutputs(
        temperature_norm=temp01_full,
        hotspot_mask=mask_full,
        overlay_rgb=overlay_img,
        stats=stats,
    )
