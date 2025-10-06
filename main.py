#!/usr/bin/env python3
"""Thermal Delamination Detector GUI.

This script provides a desktop workflow for extracting temperature data from
radiometric JPEG (RJPG) files captured by thermal drones, detecting thermal
anomalies consistent with subsurface delaminations, and exporting annotated
imagery ready for downstream orthomosaic and GIS analysis.  It uses
CustomTkinter for an accessible dark themed interface and supports drag and
drop via ``tkinterdnd2``.

Supported inputs
----------------
* DJI aircraft that embed radiometric data (via ``thermal_parser``)
* FLIR / Skydio RJPGs (via ``flirimageextractor`` – requires ExifTool in PATH)

Processing steps
----------------
1. Extract radiometric temperature data as ``float`` Celsius arrays.
2. Normalize the temperature distribution to ``0-1`` for visualization.
3. Apply a configurable statistical threshold (``mean + k * std``).
4. Clean the binary mask via morphological opening and closing with an
   adjustable kernel size.
5. Overlay hotspot pixels in red on top of a Jet colormap render of the
   thermal data while keeping the original EXIF metadata intact.

The resulting optimized JPEGs are saved inside an ``optimized`` directory by
default so they can be ingested by orthomosaic software such as OpenDroneMap,
then later converted into vector deliverables with GIS tooling like PyQGIS.

The application is intentionally rule based and does not rely on machine
learning models so that it remains transparent and easy to audit.
"""
from __future__ import annotations

import math
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

try:
    from tkinter import filedialog, messagebox  # type: ignore
except Exception as exc:  # pragma: no cover - fallback for unusual Tk builds
    raise ImportError("Tkinter is required for this application") from exc

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception as exc:  # pragma: no cover - required dependency
    raise ImportError(
        "tkinterdnd2 must be installed to provide drag-and-drop support"
    ) from exc

try:
    from CTkToolTip import CTkToolTip
except Exception as exc:  # pragma: no cover - required dependency
    raise ImportError("CTkToolTip must be installed for help tooltips") from exc

try:  # Optional – only required when DJI radiometric files are processed
    from thermal_parser import ThermalParser  # type: ignore
except Exception:  # pragma: no cover - allow program to run without it
    ThermalParser = None  # type: ignore

try:  # Optional – only required when FLIR / Skydio radiometric files are processed
    from flirimageextractor import FlirImageExtractor  # type: ignore
except Exception:  # pragma: no cover - allow program to run without it
    FlirImageExtractor = None  # type: ignore

import cv2
import piexif
from skimage.morphology import closing, opening, square

APP_TITLE = "Thermal Delamination Detector"
SUPPORTED_EXTENSIONS = {".rjpg", ".jpg", ".jpeg"}
DEFAULT_OUTPUT_DIRNAME = "optimized"

DRONE_TYPES = (
    "Auto detect",
    "DJI (thermal_parser)",
    "FLIR/Skydio (flirimageextractor)",
)


@dataclass
class ProcessingConfig:
    """Configuration parameters for the detection workflow."""

    threshold_multiplier: float = 1.5
    kernel_size: int = 3
    drone_type: str = DRONE_TYPES[0]


@dataclass
class ThermalProcessingResult:
    """Container with processed thermal data for previews and export."""

    base_image: Image.Image
    overlay_image: Image.Image
    mask: np.ndarray
    temperature_c: np.ndarray
    threshold_c: float


def _load_radiometric_from_dji(path: Path) -> np.ndarray:
    if ThermalParser is None:
        raise RuntimeError(
            "thermal_parser is not installed. Install it to process DJI RJPG files."
        )
    parser = ThermalParser(str(path))
    candidates = [
        name
        for name in (
            "get_thermal_np",
            "get_radiometric_np",
            "extract_thermal_np",
            "extract_temperature",
        )
        if hasattr(parser, name)
    ]
    if not candidates:
        raise RuntimeError(
            "thermal_parser.ThermalParser does not expose a known method to "
            "retrieve temperature data. Update the library or check its API."
        )
    last_exc: Optional[Exception] = None
    for attr in candidates:
        func = getattr(parser, attr)
        try:
            data = func()
            arr = np.asarray(data, dtype=float)
            if arr.size == 0:
                raise ValueError("Empty thermal array returned from thermal_parser")
            return arr
        except Exception as exc:  # pragma: no cover - depends on external lib
            last_exc = exc
    if last_exc is not None:
        raise RuntimeError(
            f"Failed to decode DJI radiometric data using thermal_parser: {last_exc}"
        ) from last_exc
    raise RuntimeError("Unknown error retrieving DJI radiometric data")


def _load_radiometric_from_flir(path: Path) -> np.ndarray:
    if FlirImageExtractor is None:
        raise RuntimeError(
            "flirimageextractor is not installed. Install it to process FLIR/Skydio files."
        )
    extractor = FlirImageExtractor()
    try:
        extractor.process_image(str(path))  # pragma: no cover - depends on external lib
        arr = extractor.get_thermal_np()
    except Exception as exc:  # pragma: no cover - depends on external lib
        raise RuntimeError(
            "Failed to decode FLIR radiometric data. Ensure ExifTool is installed "
            "and available on your PATH."
        ) from exc
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        raise RuntimeError("Empty thermal array returned from flirimageextractor")
    return arr


def load_temperature_array(path: Path, drone_type: str) -> np.ndarray:
    """Load a temperature array (°C) from a radiometric JPEG."""

    errors: List[str] = []
    choices: Sequence[str]
    if drone_type == DRONE_TYPES[0]:
        choices = (DRONE_TYPES[1], DRONE_TYPES[2])
    else:
        choices = (drone_type,)

    for choice in choices:
        try:
            if choice == DRONE_TYPES[1]:
                return _load_radiometric_from_dji(path)
            if choice == DRONE_TYPES[2]:
                return _load_radiometric_from_flir(path)
        except Exception as exc:
            errors.append(f"{choice}: {exc}")
    raise RuntimeError(
        "Could not extract radiometric data.\n" + "\n".join(errors)
    )


def normalise_temperature(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Return normalized 0-1 array along with min/max values."""

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise RuntimeError("Radiometric data does not contain valid temperature values")
    t_min = float(np.min(finite))
    t_max = float(np.max(finite))
    if math.isclose(t_min, t_max):
        norm = np.zeros_like(arr, dtype=float)
    else:
        norm = (arr - t_min) / (t_max - t_min)
    norm = np.clip(norm, 0.0, 1.0)
    return norm, t_min, t_max


def detect_hotspots(
    temperature: np.ndarray, multiplier: float, kernel_size: int
) -> Tuple[np.ndarray, float]:
    """Return cleaned hotspot mask and threshold temperature."""

    finite = temperature[np.isfinite(temperature)]
    if finite.size == 0:
        raise RuntimeError("No valid pixels available for thresholding")
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    threshold = mean + multiplier * std
    mask = np.greater(temperature, threshold, where=np.isfinite(temperature))

    if kernel_size > 1:
        kernel = square(kernel_size)
        mask = opening(mask, kernel)
        mask = closing(mask, kernel)
    return mask.astype(bool), threshold


def render_overlay(
    temperature_norm: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Create display-ready RGB arrays for the thermal render and overlay."""

    temp_uint8 = np.clip(temperature_norm * 255, 0, 255).astype(np.uint8)
    jet_bgr = cv2.applyColorMap(temp_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
    overlay = jet_rgb.copy()
    if np.any(mask):
        hotspot_color = np.array([255, 0, 0], dtype=np.uint8)
        overlay_mask = mask[..., None].astype(np.float32)
        base = overlay.astype(np.float32)
        overlay = np.where(
            overlay_mask > 0,
            np.clip(0.4 * base + 0.6 * hotspot_color, 0, 255),
            base,
        ).astype(np.uint8)
    return jet_rgb, overlay


def preserve_exif_and_save(src_path: Path, rgb_array: np.ndarray, out_path: Path) -> None:
    """Save RGB array as JPEG while preserving EXIF metadata via piexif."""

    image = Image.fromarray(rgb_array)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="JPEG", quality=95, subsampling=0)
    try:
        exif_dict = piexif.load(str(src_path))
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(out_path))
    except Exception:
        # Preserve best effort. If EXIF is missing or piexif fails we leave the JPEG as-is
        traceback.print_exc()


def process_rjpg(path: Path, config: ProcessingConfig) -> ThermalProcessingResult:
    """Process a single RJPG path and return results for preview/export."""

    temperature = load_temperature_array(path, config.drone_type)
    norm, _, _ = normalise_temperature(temperature)
    mask, threshold = detect_hotspots(temperature, config.threshold_multiplier, config.kernel_size)
    jet_rgb, overlay_rgb = render_overlay(norm, mask)

    try:
        base_image = Image.open(path)
    except Exception:
        # Fall back to rendering the jet map if the JPEG cannot be decoded
        base_image = Image.fromarray(jet_rgb)
    overlay_image = Image.fromarray(overlay_rgb)
    return ThermalProcessingResult(
        base_image=base_image,
        overlay_image=overlay_image,
        mask=mask,
        temperature_c=temperature,
        threshold_c=threshold,
    )


def list_candidate_files(path: Path) -> List[Path]:
    """Return a sorted list of supported RJPG files for processing."""

    if path.is_file():
        return [path]
    if not path.exists():
        return []
    files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(path.glob(f"*{ext}"))
        files.extend(path.glob(f"*{ext.upper()}"))
    return sorted({f.resolve() for f in files})


class ThermalDelamApp(TkinterDnD.TkinterDnD, ctk.CTk):
    """CustomTkinter GUI application for radiometric thermal processing."""

    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        TkinterDnD.TkinterDnD.__init__(self)
        ctk.CTk.__init__(self)

        self.title(APP_TITLE)
        self.geometry("1100x720")
        self.minsize(1000, 680)

        self.config = ProcessingConfig()
        self.current_path: Optional[Path] = None
        self.preview_photo_original: Optional[ImageTk.PhotoImage] = None
        self.preview_photo_processed: Optional[ImageTk.PhotoImage] = None
        self.preview_pending = False

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = ctk.CTkFrame(self)
        root.pack(fill=ctk.BOTH, expand=True, padx=16, pady=16)

        # Input controls -------------------------------------------------
        input_frame = ctk.CTkFrame(root)
        input_frame.pack(fill=ctk.X, padx=10, pady=(10, 5))

        self.input_entry = ctk.CTkEntry(input_frame, placeholder_text="Drag & drop a file or folder here")
        self.input_entry.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(10, 6), pady=10)
        self.input_entry.drop_target_register(DND_FILES)
        self.input_entry.dnd_bind("<Drop>", self._on_drop)
        CTkToolTip(self.input_entry, message="Input RJPG file or folder containing radiometric JPEGs")

        browse_btn = ctk.CTkButton(input_frame, text="Browse…", command=self._select_input)
        browse_btn.pack(side=ctk.LEFT, padx=(0, 10), pady=10)
        CTkToolTip(browse_btn, message="Open a file/folder picker to choose RJPG inputs")

        output_frame = ctk.CTkFrame(root)
        output_frame.pack(fill=ctk.X, padx=10, pady=5)

        self.output_entry = ctk.CTkEntry(output_frame, placeholder_text="Output folder (defaults to <input>/optimized)")
        self.output_entry.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(10, 6), pady=10)
        CTkToolTip(self.output_entry, message="Folder where optimized overlays will be exported")

        output_btn = ctk.CTkButton(output_frame, text="Choose output…", command=self._select_output)
        output_btn.pack(side=ctk.LEFT, padx=(0, 10), pady=10)
        CTkToolTip(output_btn, message="Select a different output folder if desired")

        # Parameter frame ------------------------------------------------
        param_frame = ctk.CTkFrame(root)
        param_frame.pack(fill=ctk.X, padx=10, pady=(5, 10))

        drone_label = ctk.CTkLabel(param_frame, text="Drone platform")
        drone_label.grid(row=0, column=0, padx=10, pady=(12, 6), sticky="w")
        CTkToolTip(drone_label, message="Choose Auto, DJI, or FLIR/Skydio decoding")

        self.drone_menu = ctk.CTkOptionMenu(
            param_frame,
            values=list(DRONE_TYPES),
            command=self._on_drone_change,
        )
        self.drone_menu.set(self.config.drone_type)
        self.drone_menu.grid(row=0, column=1, padx=10, pady=(12, 6), sticky="ew")
        CTkToolTip(self.drone_menu, message="Select decoding path that matches your dataset")

        thresh_label = ctk.CTkLabel(param_frame, text="Threshold multiplier (σ)")
        thresh_label.grid(row=1, column=0, padx=10, pady=6, sticky="w")
        CTkToolTip(thresh_label, message="Higher values detect fewer, more extreme hotspots")

        slider_frame = ctk.CTkFrame(param_frame)
        slider_frame.grid(row=1, column=1, padx=10, pady=6, sticky="ew")
        slider_frame.columnconfigure(0, weight=1)

        self.threshold_slider = ctk.CTkSlider(
            slider_frame,
            from_=1.0,
            to=3.0,
            number_of_steps=40,
            command=self._on_threshold_change,
        )
        self.threshold_slider.set(self.config.threshold_multiplier)
        self.threshold_slider.grid(row=0, column=0, sticky="ew", padx=(4, 4), pady=(8, 4))
        CTkToolTip(self.threshold_slider, message="Adjust the statistical cut-off applied to temperatures")

        self.threshold_value_label = ctk.CTkLabel(slider_frame, text=f"{self.config.threshold_multiplier:.2f} σ")
        self.threshold_value_label.grid(row=0, column=1, padx=(6, 4))

        kernel_label = ctk.CTkLabel(param_frame, text="Morphology kernel (px)")
        kernel_label.grid(row=2, column=0, padx=10, pady=(6, 12), sticky="w")
        CTkToolTip(kernel_label, message="Increase to smooth and fill hotspot clusters")

        kernel_frame = ctk.CTkFrame(param_frame)
        kernel_frame.grid(row=2, column=1, padx=10, pady=(6, 12), sticky="ew")
        kernel_frame.columnconfigure(0, weight=1)

        self.kernel_slider = ctk.CTkSlider(
            kernel_frame,
            from_=1,
            to=10,
            number_of_steps=9,
            command=self._on_kernel_change,
        )
        self.kernel_slider.set(self.config.kernel_size)
        self.kernel_slider.grid(row=0, column=0, sticky="ew", padx=(4, 4), pady=(8, 4))
        CTkToolTip(self.kernel_slider, message="Controls the opening/closing structuring element size")

        self.kernel_value_label = ctk.CTkLabel(kernel_frame, text=f"{self.config.kernel_size:d} px")
        self.kernel_value_label.grid(row=0, column=1, padx=(6, 4))

        # Buttons --------------------------------------------------------
        button_frame = ctk.CTkFrame(root)
        button_frame.pack(fill=ctk.X, padx=10, pady=(0, 10))

        self.preview_button = ctk.CTkButton(button_frame, text="Update Preview", command=self._update_preview)
        self.preview_button.pack(side=ctk.LEFT, padx=(10, 6), pady=10)
        CTkToolTip(self.preview_button, message="Apply the current parameters to the selected file")

        self.process_button = ctk.CTkButton(button_frame, text="Process All", command=self._process_all)
        self.process_button.pack(side=ctk.LEFT, padx=(6, 10), pady=10)
        CTkToolTip(self.process_button, message="Batch process every supported RJPG in the selection")

        # Preview section ------------------------------------------------
        preview_frame = ctk.CTkFrame(root)
        preview_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=(0, 10))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        preview_label_original = ctk.CTkLabel(preview_frame, text="Original RJPG", anchor="center")
        preview_label_original.grid(row=0, column=0, padx=10, pady=(10, 4))
        CTkToolTip(preview_label_original, message="Raw JPEG as stored on the aircraft")

        preview_label_processed = ctk.CTkLabel(preview_frame, text="Thermal overlay", anchor="center")
        preview_label_processed.grid(row=0, column=1, padx=10, pady=(10, 4))
        CTkToolTip(preview_label_processed, message="Jet render with red highlight overlay")

        self.original_canvas = ctk.CTkLabel(preview_frame, text="Drop RJPG to preview", anchor="center")
        self.original_canvas.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.original_canvas.configure(compound="center")
        CTkToolTip(self.original_canvas, message="Preview of the dropped radiometric JPEG")

        self.processed_canvas = ctk.CTkLabel(preview_frame, text="Processed overlay preview", anchor="center")
        self.processed_canvas.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.processed_canvas.configure(compound="center")
        CTkToolTip(self.processed_canvas, message="Thermal overlay result with hotspots highlighted")

        # Status/logging -------------------------------------------------
        log_frame = ctk.CTkFrame(root)
        log_frame.pack(fill=ctk.BOTH, expand=False, padx=10, pady=(0, 10))

        self.progress = ctk.CTkProgressBar(log_frame)
        self.progress.pack(fill=ctk.X, padx=10, pady=(10, 6))
        self.progress.set(0)
        CTkToolTip(self.progress, message="Shows batch processing progress")

        self.log_text = ctk.CTkTextbox(log_frame, height=120)
        self.log_text.pack(fill=ctk.BOTH, expand=True, padx=10, pady=(0, 10))
        CTkToolTip(self.log_text, message="Processing messages and helpful reminders")
        self._log_best_practice_notes()

    # ------------------------------------------------------------------ Helpers
    def _on_drop(self, event) -> None:
        data = event.data
        paths = self.tk.splitlist(data)
        if not paths:
            return
        path = Path(paths[0])
        if not path.exists():
            messagebox.showerror(APP_TITLE, f"Dropped path not found: {path}")
            return
        self._set_input_path(path)

    def _select_input(self) -> None:
        path = filedialog.askopenfilename(title="Select RJPG file", filetypes=[("Radiometric JPEG", "*.rjpg"), ("JPEG", "*.jpg"), ("All", "*.*")])
        if not path:
            directory = filedialog.askdirectory(title="Select folder with RJPG files")
            if not directory:
                return
            self._set_input_path(Path(directory))
            return
        self._set_input_path(Path(path))

    def _select_output(self) -> None:
        directory = filedialog.askdirectory(title="Select output folder")
        if not directory:
            return
        self.output_entry.delete(0, ctk.END)
        self.output_entry.insert(0, directory)

    def _set_input_path(self, path: Path) -> None:
        self.current_path = path
        self.input_entry.delete(0, ctk.END)
        self.input_entry.insert(0, str(path))
        if not self.output_entry.get():
            default_out = (path.parent if path.is_file() else path) / DEFAULT_OUTPUT_DIRNAME
            self.output_entry.insert(0, str(default_out))
        self._update_preview()

    def _on_drone_change(self, value: str) -> None:
        self.config.drone_type = value
        self._schedule_preview_refresh()

    def _on_threshold_change(self, value: float) -> None:
        self.config.threshold_multiplier = float(value)
        self.threshold_value_label.configure(text=f"{float(value):.2f} σ")
        self._schedule_preview_refresh()

    def _on_kernel_change(self, value: float) -> None:
        self.config.kernel_size = max(1, int(round(value)))
        self.kernel_value_label.configure(text=f"{self.config.kernel_size:d} px")
        self._schedule_preview_refresh()

    def _schedule_preview_refresh(self) -> None:
        if self.preview_pending:
            return
        self.preview_pending = True
        self.after(300, self._update_preview)

    def _update_preview(self) -> None:
        self.preview_pending = False
        if not self.current_path:
            return
        files = list_candidate_files(self.current_path)
        if not files:
            messagebox.showwarning(APP_TITLE, "No RJPG files found to preview.")
            return
        target = files[0]
        try:
            result = process_rjpg(target, self.config)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Preview failed: {exc}")
            self.log(f"Preview failed for {target.name}: {exc}")
            return
        self._display_preview(result)
        self.log(
            f"Previewed {target.name}: threshold {result.threshold_c:.2f} °C, hotspots {np.count_nonzero(result.mask)} px"
        )

    def _display_preview(self, result: ThermalProcessingResult) -> None:
        self._show_image_on_label(result.base_image, self.original_canvas)
        self._show_image_on_label(result.overlay_image, self.processed_canvas)

    def _show_image_on_label(self, image: Image.Image, label: ctk.CTkLabel) -> None:
        max_w = label.winfo_width() or 480
        max_h = label.winfo_height() or 320
        img = image.copy()
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo, text="")
        if label is self.original_canvas:
            self.preview_photo_original = photo
        else:
            self.preview_photo_processed = photo

    def _process_all(self) -> None:
        if not self.current_path:
            messagebox.showerror(APP_TITLE, "Select a file or folder to process first.")
            return
        files = list_candidate_files(self.current_path)
        if not files:
            messagebox.showerror(APP_TITLE, "No RJPG files found in the selection.")
            return
        output_dir = self.output_entry.get().strip()
        if not output_dir:
            default_out = (self.current_path.parent if self.current_path.is_file() else self.current_path) / DEFAULT_OUTPUT_DIRNAME
            output_dir = str(default_out)
            self.output_entry.insert(0, output_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        self.process_button.configure(state=ctk.DISABLED)
        self.preview_button.configure(state=ctk.DISABLED)
        self.progress.set(0)
        self.log(f"Processing {len(files)} file(s) → {out_path}")

        thread = threading.Thread(
            target=self._process_worker,
            args=(files, out_path),
            daemon=True,
        )
        thread.start()

    def _process_worker(self, files: Sequence[Path], out_path: Path) -> None:
        total = len(files)
        for idx, path in enumerate(files, start=1):
            try:
                result = process_rjpg(path, self.config)
                overlay_rgb = np.array(result.overlay_image)
                output_file = out_path / f"{path.stem}_optimized.jpg"
                preserve_exif_and_save(path, overlay_rgb, output_file)
                self.log(f"Saved {output_file.name}")
            except Exception as exc:
                self.log(f"Error on {path.name}: {exc}\n{traceback.format_exc()}")
            progress_value = idx / total
            self.after(0, lambda value=progress_value: self.progress.set(value))
        self.after(0, self._on_processing_complete)

    def _on_processing_complete(self) -> None:
        self.process_button.configure(state=ctk.NORMAL)
        self.preview_button.configure(state=ctk.NORMAL)
        self.log("Processing complete. Files ready for orthomosaic workflows.")
        messagebox.showinfo(APP_TITLE, "Processing complete")

    def log(self, message: str) -> None:
        self.log_text.configure(state=ctk.NORMAL)
        self.log_text.insert(ctk.END, message + "\n")
        self.log_text.see(ctk.END)
        self.log_text.configure(state=ctk.NORMAL)

    def _log_best_practice_notes(self) -> None:
        notes = (
            "• Best practice: fly within 2-3 hours after sunrise or before sunset for maximum thermal contrast.\n"
            "• Ensure dry deck conditions and light wind (<10 mph) to avoid false positives.\n"
            "• Capture nadir imagery at consistent altitude; overlap ≥70% aids orthomosaic alignment.\n"
            "• Record visible RGB alongside thermal for contextual inspection and verification.\n"
        )
        self.log_text.insert(ctk.END, notes)
        self.log_text.configure(state=ctk.NORMAL)


def run() -> None:
    app = ThermalDelamApp()
    app.mainloop()


if __name__ == "__main__":
    run()
