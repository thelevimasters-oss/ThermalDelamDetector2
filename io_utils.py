# io_utils.py
"""
IO utilities for Thermal Delamination Detector.

- Discover images in a folder (optionally recursive)
- Save images with EXIF preservation
- Export temperature arrays and masks as images/CSV/JSON
- Write a PDF report via ReportLab (if available)
- Logging setup
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

SUPPORTED_EXTS = {".rjpg", ".jpg", ".jpeg", ".tif", ".tiff"}


def ensure_logging(log_path: Optional[str | Path] = None) -> logging.Logger:
    """
    Create a rotating logger writing to file and console.
    """
    logger = logging.getLogger("thermal_delam")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_path:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def discover_images(folder: str | Path, recursive: bool = False) -> List[Path]:
    p = Path(folder)
    if not p.is_dir():
        return []
    if recursive:
        files = [f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTS and f.is_file()]
    else:
        files = [f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTS and f.is_file()]
    return sorted(files)


def save_jpeg_with_exif(img: Image.Image, dest_path: str | Path, source_img: Optional[Image.Image] = None, quality: int = 95) -> None:
    """
    Save JPEG preserving EXIF if available from source_img or img.info['exif'].
    """
    exif = None
    if source_img is not None:
        exif = source_img.info.get("exif")
    if exif is None:
        exif = img.info.get("exif")
    img.save(str(dest_path), format="JPEG", quality=int(quality), exif=exif)


def save_png(img: Image.Image, dest_path: str | Path) -> None:
    img.save(str(dest_path), format="PNG")


def save_mask_png(mask: np.ndarray, dest_path: str | Path) -> None:
    """
    Save boolean mask as PNG (white=hotspot, black=background).
    """
    img = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    img.save(str(dest_path), format="PNG")


def save_temperature_png(temp01: np.ndarray, dest_path: str | Path) -> None:
    """
    Save normalized temperature as grayscale PNG.
    """
    img = Image.fromarray((np.clip(temp01, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    img.save(str(dest_path), format="PNG")


def export_temperature_csv(temp01: np.ndarray, dest_path: str | Path) -> None:
    """
    Export normalized temperature map as CSV. For large images this can be big.
    """
    h, w = temp01.shape
    with open(dest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# normalized_temperature_rows", h, "cols", w])
        for y in range(h):
            row = temp01[y, :].astype(np.float32).round(6).tolist()
            writer.writerow(row)


def export_stats_json(stats: dict, dest_path: str | Path) -> None:
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def make_output_paths(
    input_path: str | Path,
    output_dir: str | Path,
    suffix: str,
) -> Path:
    """
    Build output path with *suffix* appended to the stem.

    If *suffix* already contains a file extension (e.g. ``"_overlay.jpg"`` or
    ``".json"``) it will be used as-is. Otherwise the source image extension is
    preserved.
    """
    in_p = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix_str = str(suffix)
    suffix_ext = Path(suffix_str).suffix
    if suffix_ext:
        filename = f"{in_p.stem}{suffix_str}"
    else:
        filename = f"{in_p.stem}{suffix_str}{in_p.suffix}"

    return out_dir / filename


def write_pdf_report(
    input_file: str | Path,
    overlay_jpeg_path: str | Path,
    stats: dict,
    pdf_path: str | Path,
    thumbs: Optional[Sequence[Image.Image]] = None,
) -> None:
    """
    Generate a simple PDF report with stats and thumbnails using ReportLab if available.
    """
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore
        from reportlab.lib.utils import ImageReader  # type: ignore
    except Exception as e:
        raise RuntimeError("ReportLab is required for PDF export but not installed.") from e

    pdf_path = str(pdf_path)
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 50, "Thermal Delamination Detector Report")
    c.setFont("Helvetica", 10)
    c.drawString(40, height - 65, f"Source file: {Path(input_file).name}")
    c.drawString(40, height - 80, f"Generated: {datetime.now().isoformat(timespec='seconds')}")

    # Stats
    y = height - 120
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Statistics")
    y -= 16
    c.setFont("Helvetica", 10)
    for k, v in stats.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 14
        if y < 150:
            c.showPage()
            y = height - 60

    # Images
    def draw_image(path_or_img, x, y, max_w, max_h):
        if isinstance(path_or_img, (str, os.PathLike)):
            pil_img = Image.open(path_or_img)
        else:
            pil_img = path_or_img
        iw, ih = pil_img.size
        scale = min(max_w / iw, max_h / ih)
        c.drawImage(ImageReader(pil_img), x, y, iw * scale, ih * scale)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, 140, "Overlay")
    draw_image(overlay_jpeg_path, 40, 180, width - 80, height / 2.5)

    # Thumbnails (optional)
    if thumbs:
        x, y = 40, 120
        for t in thumbs:
            draw_image(t, x, 20, (width - 100) / 3, 90)
            x += (width - 100) / 3 + 10
            if x > width - 100:
                x = 40
                y -= 110

    c.showPage()
    c.save()
