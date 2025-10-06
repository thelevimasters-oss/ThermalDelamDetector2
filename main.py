# main.py
"""
Entry point for Thermal Delamination Detector.

- Ensures required dependencies (Pillow, NumPy, PyQt6, ReportLab) are installed
- Provides CLI for headless batch/single processing
- Launches PyQt6 GUI by default unless --cli is specified or no display is available

Usage (CLI):
    python main.py --input image.jpg --output outdir
    python main.py --input_folder infolder --output_folder outfolder --recursive
    python main.py --cli --help

Python 3.10+
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Dependency bootstrap
REQUIRED = ["Pillow", "numpy"]
OPTIONAL_GUI = ["PyQt6"]
OPTIONAL_PDF = ["reportlab"]

def _pip_install(pkg: str):
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", pkg], check=False)

def ensure_deps():
    missing = []
    for m in REQUIRED:
        try:
            __import__(m if m != "Pillow" else "PIL")
        except Exception:
            missing.append(m)
    for m in OPTIONAL_GUI + OPTIONAL_PDF:
        try:
            __import__(m)
        except Exception:
            # Mark optional missing (we'll install to meet features)
            missing.append(m)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        for m in missing:
            try:
                _pip_install(m)
            except Exception as ex:
                print(f"Failed to install {m}: {ex}", file=sys.stderr)

ensure_deps()

# Now import project modules
import numpy as np
from PIL import Image

from processing import ProcessingParams, process_image, SUPPORTED_EXTS
from io_utils import (
    discover_images,
    ensure_logging,
    export_stats_json,
    export_temperature_csv,
    make_output_paths,
    save_jpeg_with_exif,
    save_mask_png,
    save_temperature_png,
    write_pdf_report,
)

def has_display() -> bool:
    if platform.system() == "Linux":
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    # On macOS/Windows assume display available
    return True

def run_cli(args):
    logger = ensure_logging(Path.cwd() / "processing.log")
    params = ProcessingParams(
        hotspot_percentile=args.hotspot_percentile,
        min_cluster_size=args.min_cluster_size,
        opening_iterations=args.opening_iterations,
        closing_iterations=args.closing_iterations,
        kernel_size=args.kernel_size,
        overlay_opacity=args.overlay_opacity,
        gaussian_sigma=args.gaussian_sigma,
        colormap=args.colormap,
        use_gpu=args.gpu,
    ).clamped()

    outdir = Path(args.output or args.output_folder or Path.cwd())
    outdir.mkdir(parents=True, exist_ok=True)

    files = []
    if args.input:
        files = [Path(args.input)]
    elif args.input_folder:
        files = discover_images(args.input_folder, recursive=args.recursive)
    else:
        print("No input provided. Use --input or --input_folder.", file=sys.stderr)
        sys.exit(2)

    if not files:
        print("No supported images found.", file=sys.stderr)
        sys.exit(1)

    n_ok = 0
    for p in files:
        try:
            img = Image.open(str(p)); img.load()
            outputs = process_image(img, params)
            # Save overlay JPEG
            out_overlay = outdir / f"{p.stem}_overlay.jpg"
            save_jpeg_with_exif(outputs.overlay_rgb, out_overlay, source_img=img)
            # Save extras
            save_mask_png(outputs.hotspot_mask, outdir / f"{p.stem}_mask.png")
            save_temperature_png(outputs.temperature_norm, outdir / f"{p.stem}_temp.png")
            export_temperature_csv(outputs.temperature_norm, outdir / f"{p.stem}_temp.csv")
            export_stats_json(outputs.stats, outdir / f"{p.stem}_stats.json")

            if args.report_pdf:
                try:
                    temp_img = Image.fromarray((np.clip(outputs.temperature_norm, 0, 1) * 255).astype(np.uint8), "L").convert("RGB")
                    mask_img = Image.fromarray(np.where(outputs.hotspot_mask, 255, 0).astype(np.uint8), "L").convert("RGB")
                    write_pdf_report(str(p), out_overlay, outputs.stats, outdir / f"{p.stem}_report.pdf", thumbs=[img.convert("RGB"), temp_img, mask_img])
                except Exception as ex:
                    logger.warning(f"PDF report failed for {p}: {ex}")
            logger.info(f"Processed {p.name}: hotspots={outputs.stats['hotspot_pixels']} ({outputs.stats['hotspot_coverage_percent']:.2f}%)")
            n_ok += 1
        except Exception as ex:
            logger.exception(f"Failed to process {p}: {ex}")

    print(f"Done. {n_ok}/{len(files)} processed. Outputs in: {outdir}")

def build_parser():
    ap = argparse.ArgumentParser(description="Thermal Delamination Detector")
    ap.add_argument("--cli", action="store_true", help="Force CLI mode")
    ap.add_argument("--input", type=str, help="Single input image path")
    ap.add_argument("--output", type=str, help="Output folder for single image")
    ap.add_argument("--input_folder", type=str, help="Input folder (batch)")
    ap.add_argument("--output_folder", type=str, help="Output folder (batch)")
    ap.add_argument("--recursive", action="store_true", help="Process subfolders recursively")

    ap.add_argument("--hotspot_percentile", type=float, default=97.0)
    ap.add_argument("--min_cluster_size", type=int, default=45)
    ap.add_argument("--opening_iterations", type=int, default=1)
    ap.add_argument("--closing_iterations", type=int, default=1)
    ap.add_argument("--kernel_size", type=int, default=3)
    ap.add_argument("--overlay_opacity", type=float, default=0.6)
    ap.add_argument("--gaussian_sigma", type=float, default=0.0)
    ap.add_argument("--colormap", type=str, default="blue_red", choices=["blue_red", "jet", "viridis"])
    ap.add_argument("--gpu", action="store_true", help="Use GPU via PyTorch if available")
    ap.add_argument("--report_pdf", action="store_true", help="Also generate a PDF report")
    return ap

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cli or not has_display():
        run_cli(args)
    else:
        try:
            from gui import run_gui  # type: ignore
        except Exception as ex:
            print(f"GUI unavailable ({ex}), falling back to CLI.", file=sys.stderr)
            run_cli(args)
            return
        run_gui()

if __name__ == "__main__":
    main()
