# Thermal Delamination Detector

A standalone CustomTkinter application for bridge deck inspectors to convert
radiometric JPEGs (RJPG) from thermal drones into visually enhanced overlays
highlighting suspected delaminations. The workflow is rule-based and does not
rely on machine learning, making it transparent and auditable for
infrastructure owners.

## Key features

- Drag-and-drop or browse radiometric JPEGs from DJI, FLIR, or Skydio drones.
- Automatic temperature extraction (``thermal_parser`` for DJI, ``flirimageextractor`` for FLIR/Skydio).
- Adjustable statistical threshold (``mean + k × σ``) and morphology kernel for hotspot cleaning.
- Dual preview panes showing the original RJPG and the Jet-colormap overlay with red highlighted hotspots.
- Batch processing with progress feedback and EXIF preservation so exports remain geotagged for orthomosaic/GIS workflows.
- Built-in best practice reminders for capturing reliable thermal data on bridge decks.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # optional convenience, see list below
```

Required Python packages (install via ``pip install <name>``):

- ``customtkinter``
- ``tkinterdnd2``
- ``flirimageextractor``
- ``thermal-parser``
- ``numpy``
- ``scikit-image``
- ``opencv-python-headless``
- ``piexif``
- ``Pillow`` (installed automatically with CustomTkinter but listed for clarity)
- ``CTkToolTip``

> **Note:** ``flirimageextractor`` depends on [ExifTool](https://exiftool.org/).
> Ensure ``exiftool`` is installed on your system and available on the ``PATH``
> when processing FLIR or Skydio RJPG files.

If you prefer not to maintain a requirements file, install the packages
manually:

```bash
pip install customtkinter tkinterdnd2 flirimageextractor thermal-parser numpy scikit-image opencv-python-headless piexif Pillow CTkToolTip
```

## Usage

```bash
python main.py
```

1. Drag a radiometric JPEG (RJPG) file or an entire folder onto the input box,
   or use the **Browse…** button.
2. Confirm the drone platform (Auto detect, DJI, or FLIR/Skydio) from the
   dropdown.
3. Adjust the **Threshold multiplier** slider (1.0–3.0, default 1.5) to tighten
   or relax hotspot detection.
4. Adjust the **Morphology kernel** slider (1–10 pixels, default 3) to smooth
   or fill hotspot regions.
5. Click **Update Preview** to see the overlay before running a batch.
6. Choose an output folder (defaults to ``optimized`` inside the input folder).
7. Press **Process All** to export optimized JPEGs ready for orthomosaic tools
   such as OpenDroneMap and subsequent GIS analysis (e.g., PyQGIS polygonization).

The application preserves all original EXIF/GPS metadata inside the exported
JPEGs, enabling seamless alignment with photogrammetry software and GIS
systems.

## Recommended field workflow

- Fly within 2–3 hours after sunrise or before sunset to maximize thermal
  contrast between intact and delaminated concrete.
- Inspect decks when dry with light winds (<10 mph) to reduce thermal noise and
  false positives.
- Capture nadir thermal imagery with ≥70% overlap; pair with RGB for context.
- Maintain consistent altitude/speed to avoid thermal drift and ensure
  orthomosaic alignment.

## Troubleshooting

- **Missing dependencies**: Re-run the installation commands above. The GUI will
  raise a descriptive error if the required decoding library is unavailable.
- **FLIR/Skydio parsing issues**: Confirm ``exiftool`` is installed and
  accessible from the command line.
- **EXIF metadata**: Outputs fall back gracefully if EXIF cannot be copied, but
  you can inspect console logs for details.

## License

This project is provided as-is for bridge inspection teams. Verify outputs with
in-field observations and follow your organization’s QA/QC procedures.
