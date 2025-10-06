# gui.py
"""
PyQt6 GUI for Thermal Delamination Detector.

Features:
- Tabs: Single Image, Batch Processing, Settings/About
- Drag-and-drop support for files/folders
- Parameter controls with live preview (original, temperature, mask, overlay)
- ROI selection via rubber band
- Undo/redo of parameter changes
- Progress bar and logging panel for batch
- Export overlay/mask/temperature, JSON/CSV, PDF report
- Light/Dark theme toggle

Dependencies: PyQt6, Pillow, NumPy, ReportLab (for PDF export optional at runtime)
"""
from __future__ import annotations

import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

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
    save_png,
    save_temperature_png,
    write_pdf_report,
)

from PyQt6 import QtCore, QtGui, QtWidgets


def _qimage_from_array(arr: np.ndarray) -> QtGui.QImage:
    """
    Convert HxW or HxWx3 ndarray to QImage.
    """
    if arr.ndim == 2:
        h, w = arr.shape
        buf = np.clip(arr, 0, 255).astype(np.uint8, copy=True)
        # QImage does not take ownership of the NumPy buffer, so we copy to detach
        qimg = QtGui.QImage(buf.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8).copy()
        qimg.setDevicePixelRatio(1)
        return qimg
    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        buf = np.clip(arr, 0, 255).astype(np.uint8, copy=True)
        qimg = QtGui.QImage(buf.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
        qimg.setDevicePixelRatio(1)
        return qimg
    else:
        raise ValueError("Unsupported array shape for preview")


class ImageView(QtWidgets.QGraphicsView):
    """
    GraphicsView with Pixmap and ROI selection using QRubberBand.
    """
    roiChanged = QtCore.pyqtSignal(tuple)  # (x, y, w, h)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self._pix = None
        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, self)
        self._origin = None
        self._roi = None
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

    def set_image(self, qimage: QtGui.QImage):
        pix = QtGui.QPixmap.fromImage(qimage)
        self.scene().clear()
        self._pix = self.scene().addPixmap(pix)
        # QGraphicsPixmapItem.boundingRect() already returns a QRectF, so use it directly
        # instead of wrapping the integer QRect from QPixmap.rect().
        self.setSceneRect(self._pix.boundingRect())
        self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.scene() and not self.scene().sceneRect().isNull():
            self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.MouseButton.LeftButton and self._pix is not None:
            self._origin = e.position().toPoint()
            self._rubber.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
            self._rubber.show()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._rubber.isVisible():
            rect = QtCore.QRect(self._origin, e.position().toPoint()).normalized()
            self._rubber.setGeometry(rect)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.MouseButton.LeftButton and self._rubber.isVisible():
            rect = self._rubber.geometry()
            self._rubber.hide()
            # Map to image coordinates
            if self._pix:
                # Map viewport rect to scene, then to image
                topLeft = self.mapToScene(rect.topLeft()).toPoint()
                bottomRight = self.mapToScene(rect.bottomRight()).toPoint()
                x, y = max(0, topLeft.x()), max(0, topLeft.y())
                w, h = max(1, bottomRight.x() - x), max(1, bottomRight.y() - y)
                self._roi = (x, y, w, h)
                self.roiChanged.emit(self._roi)
        super().mouseReleaseEvent(e)

    def clear_roi(self):
        self._roi = None
        self.roiChanged.emit((0, 0, 0, 0))

    def roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._roi


class ParamHistory:
    """
    Simple undo/redo stack for ProcessingParams.
    """
    def __init__(self, initial: ProcessingParams):
        self._stack = [initial]
        self._idx = 0

    def current(self) -> ProcessingParams:
        return self._stack[self._idx]

    def push(self, p: ProcessingParams):
        # Truncate forward history
        self._stack = self._stack[: self._idx + 1]
        self._stack.append(p)
        self._idx += 1

    def can_undo(self) -> bool:
        return self._idx > 0

    def can_redo(self) -> bool:
        return self._idx < len(self._stack) - 1

    def undo(self) -> ProcessingParams:
        if self.can_undo():
            self._idx -= 1
        return self.current()

    def redo(self) -> ProcessingParams:
        if self.can_redo():
            self._idx += 1
        return self.current()


class SingleTab(QtWidgets.QWidget):
    """
    Single Image processing tab with live preview.
    """
    def __init__(self, logger, parent=None):
        super().__init__(parent)
        self.logger = logger
        self.params = ProcessingParams()
        self.history = ParamHistory(self.params)
        self.current_img: Optional[Image.Image] = None
        self.current_path: Optional[Path] = None
        self.outputs = None  # last ProcessingOutputs

        # UI
        self._build_ui()
        self.setAcceptDrops(True)

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        # Left: Controls
        ctrl = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(ctrl)

        self.fileEdit = QtWidgets.QLineEdit()
        self.browseBtn = QtWidgets.QPushButton("Browse…")
        self.browseBtn.clicked.connect(self.on_browse)

        fileRow = QtWidgets.QHBoxLayout()
        fileRow.addWidget(self.fileEdit, 1)
        fileRow.addWidget(self.browseBtn)
        form.addRow("Input Image:", fileRow)

        self.percentSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.percentSlider.setRange(50, 100)
        self.percentSlider.setValue(int(self.params.hotspot_percentile))
        self.percentSpin = QtWidgets.QDoubleSpinBox()
        self.percentSpin.setRange(50.0, 100.0)
        self.percentSpin.setValue(self.params.hotspot_percentile)
        self.percentSpin.setDecimals(1)

        self._bind_slider_spin(self.percentSlider, self.percentSpin, "hotspot_percentile", tip="Percentile for hotspot threshold. Higher → fewer hotspots.")

        self.minSizeSpin = QtWidgets.QSpinBox()
        self.minSizeSpin.setRange(0, 1_000_000)
        self.minSizeSpin.setValue(self.params.min_cluster_size)
        self.minSizeSpin.setToolTip("Minimum hotspot cluster size (pixels).")
        self.minSizeSpin.valueChanged.connect(lambda v: self._update_param("min_cluster_size", v))

        self.openSpin = QtWidgets.QSpinBox()
        self.openSpin.setRange(0, 20)
        self.openSpin.setValue(self.params.opening_iterations)
        self.openSpin.setToolTip("Binary opening iterations (noise removal).")
        self.openSpin.valueChanged.connect(lambda v: self._update_param("opening_iterations", v))

        self.closeSpin = QtWidgets.QSpinBox()
        self.closeSpin.setRange(0, 20)
        self.closeSpin.setValue(self.params.closing_iterations)
        self.closeSpin.setToolTip("Binary closing iterations (fill small holes).")
        self.closeSpin.valueChanged.connect(lambda v: self._update_param("closing_iterations", v))

        self.kernelSpin = QtWidgets.QSpinBox()
        self.kernelSpin.setRange(1, 99)
        self.kernelSpin.setSingleStep(2)
        self.kernelSpin.setValue(self.params.kernel_size)
        self.kernelSpin.setToolTip("Morphology kernel size (odd).")
        self.kernelSpin.valueChanged.connect(lambda v: self._update_param("kernel_size", v if v % 2 == 1 else v + 1))

        self.opacitySlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacitySlider.setRange(0, 100)
        self.opacitySlider.setValue(int(self.params.overlay_opacity * 100))
        self.opacitySpin = QtWidgets.QDoubleSpinBox()
        self.opacitySpin.setRange(0.0, 1.0)
        self.opacitySpin.setSingleStep(0.01)
        self.opacitySpin.setDecimals(2)
        self.opacitySpin.setValue(self.params.overlay_opacity)
        self._bind_slider_spin(self.opacitySlider, self.opacitySpin, "overlay_opacity", scale=0.01, tip="Overlay opacity for colorized map/hotspots.")

        self.blurSpin = QtWidgets.QDoubleSpinBox()
        self.blurSpin.setRange(0.0, 20.0)
        self.blurSpin.setSingleStep(0.1)
        self.blurSpin.setValue(self.params.gaussian_sigma)
        self.blurSpin.setToolTip("Optional Gaussian blur sigma before thresholding.")
        self.blurSpin.valueChanged.connect(lambda v: self._update_param("gaussian_sigma", v))

        self.mapCombo = QtWidgets.QComboBox()
        self.mapCombo.addItems(["blue_red", "jet", "viridis"])
        self.mapCombo.setCurrentText(self.params.colormap)
        self.mapCombo.currentTextChanged.connect(lambda s: self._update_param("colormap", s))
        self.mapCombo.setToolTip("Colormap for temperature visualization.")

        self.gpuCheck = QtWidgets.QCheckBox("Enable GPU (PyTorch)")
        self.gpuCheck.setChecked(False)
        self.gpuCheck.setToolTip("Uses PyTorch if available for some operations.")
        self.gpuCheck.stateChanged.connect(lambda s: self._update_param("use_gpu", bool(s)))

        form.addRow("Hotspot Percentile:", self._hbox(self.percentSlider, self.percentSpin))
        form.addRow("Min Cluster Size:", self.minSizeSpin)
        form.addRow("Opening Iterations:", self.openSpin)
        form.addRow("Closing Iterations:", self.closeSpin)
        form.addRow("Kernel Size:", self.kernelSpin)
        form.addRow("Overlay Opacity:", self._hbox(self.opacitySlider, self.opacitySpin))
        form.addRow("Gaussian Sigma:", self.blurSpin)
        form.addRow("Colormap:", self.mapCombo)
        form.addRow("", self.gpuCheck)

        self.clearRoiBtn = QtWidgets.QPushButton("Clear ROI")
        self.clearRoiBtn.clicked.connect(lambda: self.viewOriginal.clear_roi())
        self.processBtn = QtWidgets.QPushButton("Process")
        self.processBtn.clicked.connect(self.on_process)
        self.saveOverlayBtn = QtWidgets.QPushButton("Save Overlay JPEG")
        self.saveOverlayBtn.clicked.connect(self.on_save_overlay)
        self.exportBtn = QtWidgets.QPushButton("Export (Mask/Temp/CSV/JSON/PDF)")
        self.exportBtn.clicked.connect(self.on_export)

        self.undoBtn = QtWidgets.QPushButton("Undo")
        self.undoBtn.clicked.connect(self.on_undo)
        self.redoBtn = QtWidgets.QPushButton("Redo")
        self.redoBtn.clicked.connect(self.on_redo)

        form.addRow("", self._hbox(self.clearRoiBtn, self.processBtn))
        form.addRow("", self._hbox(self.undoBtn, self.redoBtn))
        form.addRow("", self._vbox(self.saveOverlayBtn, self.exportBtn))

        layout.addWidget(ctrl, 0)

        # Right: Previews
        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        top = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.viewOriginal = ImageView()
        self.viewOriginal.setToolTip("Original image. Drag to select ROI.")
        self.viewOriginal.roiChanged.connect(lambda roi: self.statusBar.showMessage(f"ROI: {roi}"))
        self.viewTemp = ImageView()
        self.viewTemp.setToolTip("Temperature map [0..1].")
        top.addWidget(self._group("Original", self.viewOriginal))
        top.addWidget(self._group("Temperature", self.viewTemp))

        bottom = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.viewMask = ImageView()
        self.viewMask.setToolTip("Hotspot mask.")
        self.viewOverlay = ImageView()
        self.viewOverlay.setToolTip("Overlay.")
        bottom.addWidget(self._group("Mask", self.viewMask))
        bottom.addWidget(self._group("Overlay", self.viewOverlay))

        right.addWidget(top)
        right.addWidget(bottom)

        layout.addWidget(right, 1)

        # Status bar proxy
        self.statusBar = QtWidgets.QStatusBar()
        layout2 = QtWidgets.QVBoxLayout()
        layout2.addLayout(layout)
        layout2.addWidget(self.statusBar)
        self.setLayout(layout2)

    def _group(self, title: str, widget: QtWidgets.QWidget):
        group = QtWidgets.QGroupBox(title)
        v = QtWidgets.QVBoxLayout(group)
        v.addWidget(widget)
        return group

    def _hbox(self, *widgets):
        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(w)
        for a in widgets:
            l.addWidget(a)
        return w

    def _vbox(self, *widgets):
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(w)
        for a in widgets:
            l.addWidget(a)
        return w

    def _bind_slider_spin(self, slider, spin, field, scale=1.0, tip: str = ""):
        if tip:
            slider.setToolTip(tip)
            spin.setToolTip(tip)

        def on_slider(v):
            value = v * scale
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
            self._update_param(field, value)

        def on_spin(v):
            slider_value = int(round(v / scale)) if scale else int(v)
            slider_value = max(slider.minimum(), min(slider.maximum(), slider_value))
            slider.blockSignals(True)
            slider.setValue(slider_value)
            slider.blockSignals(False)
            self._update_param(field, v)

        slider.valueChanged.connect(on_slider)
        spin.valueChanged.connect(on_spin)

    def _update_param(self, name, value):
        d = asdict(self.params)
        d[name] = value
        newp = ProcessingParams(**d).clamped()
        self.params = newp
        self.history.push(newp)
        # Live preview if an image is loaded
        if self.current_img is not None:
            self.on_process(live=True)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
        urls = e.mimeData().urls()
        if not urls:
            return
        p = Path(urls[0].toLocalFile())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            self.fileEdit.setText(str(p))
            self.load_image(p)
        else:
            QtWidgets.QMessageBox.warning(self, "Unsupported", "Please drop a supported image file.")

    def on_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.rjpg *.jpg *.jpeg *.tif *.tiff)")
        if path:
            self.fileEdit.setText(path)
            self.load_image(Path(path))

    def load_image(self, p: Path):
        try:
            img = Image.open(str(p))
            img.load()
            self.current_img = img
            self.current_path = p
            # Show original
            # ImageQt.ImageQt keeps a pointer to the temporary PIL image which can
            # lead to crashes once the temporary object is garbage collected.
            # Convert to a NumPy array and build a detached QImage instead.
            rgb_arr = np.array(img.convert("RGB"))
            self.viewOriginal.set_image(_qimage_from_array(rgb_arr))
            self.statusBar.showMessage(f"Loaded {p.name} ({img.width}x{img.height})")
            self.on_process(live=True)
        except Exception as ex:
            self.logger.exception("Failed to load image")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load image:\n{ex}")

    def _set_preview(self, outputs, source_img: Image.Image):
        # Temperature
        temp8 = (np.clip(outputs.temperature_norm, 0.0, 1.0) * 255).astype(np.uint8)
        self.viewTemp.set_image(_qimage_from_array(temp8))
        # Mask
        mask8 = np.where(outputs.hotspot_mask, 255, 0).astype(np.uint8)
        self.viewMask.set_image(_qimage_from_array(mask8))
        # Overlay
        overlay_arr = np.array(outputs.overlay_rgb)
        self.viewOverlay.set_image(_qimage_from_array(overlay_arr))

    def on_process(self, live=False):
        if self.current_img is None:
            if not live:
                QtWidgets.QMessageBox.information(self, "No image", "Load an image first.")
            return
        try:
            roi = self.viewOriginal.roi()
            outputs = process_image(self.current_img, self.params, roi_xywh=roi)
            self.outputs = outputs
            self._set_preview(outputs, self.current_img)
            s = outputs.stats
            self.statusBar.showMessage(
                f"Done. Hotspots: {s['hotspot_pixels']} px ({s['hotspot_coverage_percent']:.2f}%). "
                f"T[min/avg/max]={s['temperature_min']:.3f}/{s['temperature_avg']:.3f}/{s['temperature_max']:.3f}"
            )
        except Exception as ex:
            self.logger.exception("Processing error")
            QtWidgets.QMessageBox.critical(self, "Processing error", str(ex))

    def on_save_overlay(self):
        if self.outputs is None or self.current_img is None or self.current_path is None:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Process an image first.")
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not out_dir:
            return
        out_path = Path(out_dir) / f"{self.current_path.stem}_overlay.jpg"
        try:
            save_jpeg_with_exif(self.outputs.overlay_rgb, out_path, source_img=self.current_img)
            self.statusBar.showMessage(f"Saved overlay: {out_path}")
        except Exception as ex:
            self.logger.exception("Save overlay error")
            QtWidgets.QMessageBox.critical(self, "Save error", str(ex))

    def on_export(self):
        if self.outputs is None or self.current_img is None or self.current_path is None:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "Process an image first.")
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not out_dir:
            return
        try:
            overlay_jpg = Path(out_dir) / f"{self.current_path.stem}_overlay.jpg"
            mask_png = Path(out_dir) / f"{self.current_path.stem}_mask.png"
            temp_png = Path(out_dir) / f"{self.current_path.stem}_temp.png"
            csv_path = Path(out_dir) / f"{self.current_path.stem}_temp.csv"
            json_path = Path(out_dir) / f"{self.current_path.stem}_stats.json"
            pdf_path = Path(out_dir) / f"{self.current_path.stem}_report.pdf"

            save_jpeg_with_exif(self.outputs.overlay_rgb, overlay_jpg, source_img=self.current_img)
            save_mask_png(self.outputs.hotspot_mask, mask_png)
            save_temperature_png(self.outputs.temperature_norm, temp_png)
            export_temperature_csv(self.outputs.temperature_norm, csv_path)
            export_stats_json(self.outputs.stats, json_path)
            # Thumbs: original, temp, mask
            temp_img = Image.fromarray((np.clip(self.outputs.temperature_norm, 0, 1) * 255).astype(np.uint8), "L").convert("RGB")
            mask_img = Image.fromarray(np.where(self.outputs.hotspot_mask, 255, 0).astype(np.uint8), "L").convert("RGB")
            write_pdf_report(str(self.current_path), overlay_jpg, self.outputs.stats, pdf_path, thumbs=[self.current_img.convert("RGB"), temp_img, mask_img])
            self.statusBar.showMessage(f"Exported files to {out_dir}")
        except Exception as ex:
            self.logger.exception("Export error")
            QtWidgets.QMessageBox.critical(self, "Export error", str(ex))

    def on_undo(self):
        if self.history.can_undo():
            self.params = self.history.undo()
            self._apply_params_to_ui()
            if self.current_img is not None:
                self.on_process(live=True)

    def on_redo(self):
        if self.history.can_redo():
            self.params = self.history.redo()
            self._apply_params_to_ui()
            if self.current_img is not None:
                self.on_process(live=True)

    def _apply_params_to_ui(self):
        p = self.params
        self.percentSlider.blockSignals(True)
        self.percentSpin.blockSignals(True)
        self.opacitySlider.blockSignals(True)
        self.opacitySpin.blockSignals(True)

        self.percentSlider.setValue(int(p.hotspot_percentile))
        self.percentSpin.setValue(p.hotspot_percentile)
        self.minSizeSpin.setValue(p.min_cluster_size)
        self.openSpin.setValue(p.opening_iterations)
        self.closeSpin.setValue(p.closing_iterations)
        self.kernelSpin.setValue(p.kernel_size)
        self.opacitySlider.setValue(int(p.overlay_opacity * 100))
        self.opacitySpin.setValue(p.overlay_opacity)
        self.blurSpin.setValue(p.gaussian_sigma)
        self.mapCombo.setCurrentText(p.colormap)
        self.gpuCheck.setChecked(p.use_gpu)

        self.percentSlider.blockSignals(False)
        self.percentSpin.blockSignals(False)
        self.opacitySlider.blockSignals(False)
        self.opacitySpin.blockSignals(False)


class BatchTab(QtWidgets.QWidget):
    """
    Batch processing tab with folder selection and progress.
    """
    def __init__(self, logger, parent=None):
        super().__init__(parent)
        self.logger = logger
        self.params = ProcessingParams()
        self._build_ui()
        self.setAcceptDrops(True)

    def _build_ui(self):
        v = QtWidgets.QVBoxLayout(self)

        # Paths
        pathRow = QtWidgets.QHBoxLayout()
        self.inEdit = QtWidgets.QLineEdit()
        self.inBtn = QtWidgets.QPushButton("Input Folder…")
        self.inBtn.clicked.connect(self.choose_in)
        self.recursiveCheck = QtWidgets.QCheckBox("Recursive")
        pathRow.addWidget(self.inEdit, 1)
        pathRow.addWidget(self.inBtn)
        pathRow.addWidget(self.recursiveCheck)

        outRow = QtWidgets.QHBoxLayout()
        self.outEdit = QtWidgets.QLineEdit()
        self.outBtn = QtWidgets.QPushButton("Output Folder…")
        self.outBtn.clicked.connect(self.choose_out)
        outRow.addWidget(self.outEdit, 1)
        outRow.addWidget(self.outBtn)

        v.addLayout(pathRow)
        v.addLayout(outRow)

        # Params (re-use similar controls but simpler)
        grid = QtWidgets.QGridLayout()
        self.percentSpin = QtWidgets.QDoubleSpinBox()
        self.percentSpin.setRange(50, 100)
        self.percentSpin.setValue(97.0)
        self.minSizeSpin = QtWidgets.QSpinBox(); self.minSizeSpin.setRange(0, 1_000_000); self.minSizeSpin.setValue(45)
        self.openSpin = QtWidgets.QSpinBox(); self.openSpin.setRange(0, 20); self.openSpin.setValue(1)
        self.closeSpin = QtWidgets.QSpinBox(); self.closeSpin.setRange(0, 20); self.closeSpin.setValue(1)
        self.kernelSpin = QtWidgets.QSpinBox(); self.kernelSpin.setRange(1, 99); self.kernelSpin.setSingleStep(2); self.kernelSpin.setValue(3)
        self.opacitySpin = QtWidgets.QSpinBox(); self.opacitySpin.setRange(0, 100); self.opacitySpin.setValue(60)
        self.blurSpin = QtWidgets.QDoubleSpinBox(); self.blurSpin.setRange(0.0, 20.0); self.blurSpin.setSingleStep(0.1); self.blurSpin.setValue(0.0)
        self.mapCombo = QtWidgets.QComboBox(); self.mapCombo.addItems(["blue_red", "jet", "viridis"])
        self.gpuCheck = QtWidgets.QCheckBox("GPU")

        labels = ["Percentile", "Min Cluster", "Open Iters", "Close Iters", "Kernel", "Opacity %", "Gaussian", "Colormap", "GPU"]
        widgets = [self.percentSpin, self.minSizeSpin, self.openSpin, self.closeSpin, self.kernelSpin, self.opacitySpin, self.blurSpin, self.mapCombo, self.gpuCheck]
        for i, (l, w) in enumerate(zip(labels, widgets)):
            grid.addWidget(QtWidgets.QLabel(l), i, 0)
            grid.addWidget(w, i, 1)
        v.addLayout(grid)

        # Actions
        self.startBtn = QtWidgets.QPushButton("Start Batch")
        self.startBtn.clicked.connect(self.start_batch)
        self.progress = QtWidgets.QProgressBar()
        self.logView = QtWidgets.QPlainTextEdit(); self.logView.setReadOnly(True)
        v.addWidget(self.startBtn)
        v.addWidget(self.progress)
        v.addWidget(self.logView, 1)

    def choose_in(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose input folder")
        if d: self.inEdit.setText(d)

    def choose_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d: self.outEdit.setText(d)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
        urls = e.mimeData().urls()
        if not urls: return
        p = Path(urls[0].toLocalFile())
        if p.is_dir():
            self.inEdit.setText(str(p))
        else:
            QtWidgets.QMessageBox.warning(self, "Drop a folder", "Please drop a folder to batch process.")

    def _gather_params(self) -> ProcessingParams:
        return ProcessingParams(
            hotspot_percentile=self.percentSpin.value(),
            min_cluster_size=self.minSizeSpin.value(),
            opening_iterations=self.openSpin.value(),
            closing_iterations=self.closeSpin.value(),
            kernel_size=self.kernelSpin.value(),
            overlay_opacity=self.opacitySpin.value() / 100.0,
            gaussian_sigma=self.blurSpin.value(),
            colormap=self.mapCombo.currentText(),
            use_gpu=self.gpuCheck.isChecked(),
        ).clamped()

    def log(self, msg: str):
        self.logView.appendPlainText(msg)
        self.logger.info(msg)

    def start_batch(self):
        inf = self.inEdit.text().strip()
        outf = self.outEdit.text().strip()
        if not inf or not outf:
            QtWidgets.QMessageBox.information(self, "Missing paths", "Please select input and output folders.")
            return
        images = discover_images(inf, recursive=self.recursiveCheck.isChecked())
        if not images:
            QtWidgets.QMessageBox.information(self, "No images", "No supported images found.")
            return
        params = self._gather_params()
        self.progress.setMaximum(len(images))
        self.progress.setValue(0)
        self.startBtn.setEnabled(False)

        def worker():
            ok = 0
            for i, p in enumerate(images, start=1):
                try:
                    img = Image.open(str(p)); img.load()
                    out = process_image(img, params)
                    # Save overlay & exports adjacent in output folder
                    overlay = Path(outf) / f"{p.stem}_overlay.jpg"
                    save_jpeg_with_exif(out.overlay_rgb, overlay, source_img=img)
                    save_mask_png(out.hotspot_mask, Path(outf) / f"{p.stem}_mask.png")
                    save_temperature_png(out.temperature_norm, Path(outf) / f"{p.stem}_temp.png")
                    export_temperature_csv(out.temperature_norm, Path(outf) / f"{p.stem}_temp.csv")
                    export_stats_json(out.stats, Path(outf) / f"{p.stem}_stats.json")
                    ok += 1
                    self.log(f"Processed {p.name}: hotspots={out.stats['hotspot_pixels']} ({out.stats['hotspot_coverage_percent']:.2f}%)")
                except Exception as ex:
                    self.log(f"Error processing {p}: {ex}")
                finally:
                    QtCore.QMetaObject.invokeMethod(
                        self.progress, "setValue", QtCore.Qt.ConnectionType.QueuedConnection, QtCore.Q_ARG(int, i)
                    )
            QtCore.QMetaObject.invokeMethod(
                self.startBtn, "setEnabled", QtCore.Qt.ConnectionType.QueuedConnection, QtCore.Q_ARG(bool, True)
            )
            self.log(f"Done. {ok}/{len(images)} processed.")

        threading.Thread(target=worker, daemon=True).start()


class SettingsTab(QtWidgets.QWidget):
    themeChanged = QtCore.pyqtSignal(bool)  # dark

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        self.darkCheck = QtWidgets.QCheckBox("Enable Dark Theme")
        v.addWidget(self.darkCheck)
        self.darkCheck.stateChanged.connect(lambda s: self.themeChanged.emit(bool(s)))

        about = QtWidgets.QTextBrowser()
        about.setOpenExternalLinks(True)
        about.setHtml(
            "<h3>Thermal Delamination Detector</h3>"
            "<p>Version 1.0.0</p>"
            "<p>Detect hotspots in thermal images and generate overlays, masks, and reports.</p>"
            '<p><a href="https://github.com/example/thermal-delam">GitHub Repository</a></p>'
        )
        v.addWidget(about, 1)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thermal Delamination Detector")
        self.resize(1200, 800)
        self.logger = ensure_logging(Path.cwd() / "processing.log")

        self.tabs = QtWidgets.QTabWidget()
        self.singleTab = SingleTab(self.logger)
        self.batchTab = BatchTab(self.logger)
        self.settingsTab = SettingsTab()
        self.settingsTab.themeChanged.connect(self.apply_theme)

        self.tabs.addTab(self.singleTab, "Single Image")
        self.tabs.addTab(self.batchTab, "Batch Processing")
        self.tabs.addTab(self.settingsTab, "Settings & About")

        self.setCentralWidget(self.tabs)
        self._build_menu()
        self.apply_theme(False)
        self.statusBar().showMessage("Ready")

    def _build_menu(self):
        m = self.menuBar()
        fileMenu = m.addMenu("&File")
        openAct = QtGui.QAction("Open Image…", self)
        openAct.triggered.connect(self.singleTab.on_browse)
        fileMenu.addAction(openAct)
        exitAct = QtGui.QAction("Exit", self)
        exitAct.triggered.connect(self.close)
        fileMenu.addAction(exitAct)

        helpMenu = m.addMenu("&Help")
        aboutAct = QtGui.QAction("About", self)
        aboutAct.triggered.connect(lambda: QtWidgets.QMessageBox.information(self, "About", "Thermal Delamination Detector\nVersion 1.0.0"))
        helpMenu.addAction(aboutAct)

    def apply_theme(self, dark: bool):
        if dark:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
            palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
            palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25, 25, 25))
            palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53))
            palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.GlobalColor.white)
            palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.GlobalColor.white)
            palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
            palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
            palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.white)
            palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
            palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(142, 45, 197).lighter())
            palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.black)
            self.setPalette(palette)
        else:
            self.setPalette(self.style().standardPalette())


def run_gui():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
