import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QStackedLayout,
    QScrollArea, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QRectF
from PySide6.QtGui import QPixmap, QTextCursor, QIcon
from PIL.Image import Image

import backend

class Worker(QThread):
    progress = Signal(str)
    finished = Signal(tuple)
    error = Signal(str)

    def __init__(self, sar_path, optical_path, weights_path):
        super().__init__()
        self.sar_path = sar_path
        self.optical_path = optical_path
        self.weights_path = weights_path

    def run(self):
        try:
            result_tuple = backend.run_flood_mapping_pipeline(
                sar_tif=self.sar_path,
                optical_tif=self.optical_path,
                weights_file=self.weights_path,
                progress_callback=self.progress.emit,
                output_dir=None
            )
            self.finished.emit(result_tuple)
        except Exception as e:
            self.error.emit(f"An error occurred: {e}\n\nCheck console for more details.")
            import traceback
            traceback.print_exc()

class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom_level = 0
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

    def clear_image(self):
        self._pixmap_item.setPixmap(QPixmap())
        self._scene.setSceneRect(QRectF())

    def set_image(self, pixmap):
        self.clear_image()
        self._zoom_level = 0
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._zoom_level == 0 and not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if self._pixmap_item.pixmap().isNull():
            return

        zoom_in, zoom_out = 1.25, 1 / 1.25
        if event.angleDelta().y() > 0:
            factor, self._zoom_level = zoom_in, self._zoom_level + 1
        else:
            factor, self._zoom_level = zoom_out, self._zoom_level - 1

        if self._zoom_level < 0:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._zoom_level = 0
        else:
            self.scale(factor, factor)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FloodMapper")

        self.result_image: Image | None = None
        self.legend_image: Image | None = None

        self.sar_path_edit = QLineEdit(self)
        self.sar_path_edit.setPlaceholderText("Path to SAR TIF file...")
        self.sar_path_edit.setReadOnly(True)
        self.sar_browse_btn = QPushButton("Browse...")
        self.sar_browse_btn.clicked.connect(self.browse_sar_file)

        self.optical_path_edit = QLineEdit(self)
        self.optical_path_edit.setPlaceholderText("Path to Optical TIF file...")
        self.optical_path_edit.setReadOnly(True)
        self.optical_browse_btn = QPushButton("Browse...")
        self.optical_browse_btn.clicked.connect(self.browse_optical_file)

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)

        self.save_as_btn = QPushButton("Save Results...")
        self.save_as_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        self.save_as_btn.clicked.connect(self.save_results)
        self.save_as_btn.setEnabled(False)

        self.progress_log = QTextEdit(self)
        self.progress_log.setReadOnly(True)
        self.progress_log.setLineWrapMode(QTextEdit.NoWrap)
        
        self.stats_legend_label = QLabel("Analysis statistics and legend will appear here.")
        self.stats_legend_label.setAlignment(Qt.AlignCenter)
        self.stats_legend_label.setStyleSheet("color: gray;")
        self.stats_legend_label.setWordWrap(True)

        self.stats_scroll_area = QScrollArea()
        self.stats_scroll_area.setWidgetResizable(True)
        self.stats_scroll_area.setWidget(self.stats_legend_label)
        self.stats_scroll_area.setFrameShape(QFrame.NoFrame)
        self.stats_scroll_area.setLineWidth(0)
        self.stats_scroll_area.setMidLineWidth(0)
        
        self.image_viewer = ImageViewer()
        self.image_viewer.setMinimumSize(400, 400)

        self.placeholder_label = QLabel("Output will be displayed here.")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("color: gray;")

        self.view_stack_layout = QStackedLayout()
        self.view_stack_layout.addWidget(self.image_viewer)
        self.view_stack_layout.addWidget(self.placeholder_label)

        self.view_container = QWidget()
        self.view_container.setLayout(self.view_stack_layout)
        
        self.set_border_style(active=True)

        root_layout = QVBoxLayout()

        file_selection_layout = QVBoxLayout()
        sar_layout = QHBoxLayout()
        sar_layout.addWidget(QLabel("SAR TIF:"))
        sar_layout.addWidget(self.sar_path_edit)
        sar_layout.addWidget(self.sar_browse_btn)

        optical_layout = QHBoxLayout()
        optical_layout.addWidget(QLabel("Optical TIF:"))
        optical_layout.addWidget(self.optical_path_edit)
        optical_layout.addWidget(self.optical_browse_btn)

        file_selection_layout.addLayout(sar_layout)
        file_selection_layout.addLayout(optical_layout)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.run_btn)
        controls_layout.addWidget(self.save_as_btn)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addLayout(file_selection_layout)
        left_layout.addLayout(controls_layout)
        left_layout.addWidget(QLabel("Log:"))
        left_layout.addWidget(self.progress_log, 2)
        left_layout.addWidget(QLabel("Statistics & Legend:"))
        left_layout.addWidget(self.stats_scroll_area, 3)

        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.view_container, 2)

        root_layout.addLayout(main_layout)

        container = QWidget()
        container.setLayout(root_layout)
        self.setCentralWidget(container)

        self.view_stack_layout.setCurrentWidget(self.placeholder_label)
        self.last_line_is_progress = False
        self.check_inputs()

        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(__file__)
        icon_path = os.path.join(base_path, 'logo.svg')
        self.setWindowIcon(QIcon(icon_path))

    def set_border_style(self, active: bool):
        if active:
            style = "border: 1px solid #c0c0c0; border-radius: 2px;"
        else:
            style = "border: none;"
        self.view_container.setStyleSheet(style)
        self.stats_scroll_area.viewport().setStyleSheet(style)

    def browse_sar_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SAR TIF File", "", "TIF Files (*.tif *.tiff)")
        if path:
            self.sar_path_edit.setText(path)
            self.check_inputs()

    def browse_optical_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Optical TIF File", "", "TIF Files (*.tif *.tiff)")
        if path:
            self.optical_path_edit.setText(path)
            self.check_inputs()

    def check_inputs(self):
        has_inputs = bool(self.sar_path_edit.text() and self.optical_path_edit.text())
        self.run_btn.setEnabled(has_inputs)

    def run_analysis(self):
        self.run_btn.setEnabled(False)
        self.save_as_btn.setEnabled(False)
        self.run_btn.setText("Processing...")
        self.progress_log.clear()
        
        self.set_border_style(active=True)

        self.stats_legend_label.setPixmap(QPixmap())
        self.stats_legend_label.setText("Processing, please wait…")
        self.stats_legend_label.setAlignment(Qt.AlignCenter)

        self.placeholder_label.setText("Processing, please wait…")
        self.view_stack_layout.setCurrentWidget(self.placeholder_label)
        self.image_viewer.clear_image() 
        
        self.last_line_is_progress = False
        self.result_image = None
        self.legend_image = None

        sar_path = self.sar_path_edit.text()
        optical_path = self.optical_path_edit.text()

        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(__file__)

        weights_path = os.path.join(base_path, 'SegformerJaccardLoss.pth')

        if not os.path.exists(weights_path):
            self.show_error(f"FATAL: Weights file not found!\nExpected at: {weights_path}")
            self.analysis_finished(None)
            return

        self.worker = Worker(sar_path, optical_path, weights_path)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.show_error)
        self.worker.start()

    def save_results(self):
        if not (self.result_image and self.legend_image):
            QMessageBox.warning(self, "No Results", "There are no analysis results to save.")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Results")

        if save_dir:
            try:
                base_name = os.path.splitext(os.path.basename(self.sar_path_edit.text()))[0]
                map_path = os.path.join(save_dir, f"{base_name}_flood_map.png")
                legend_path = os.path.join(save_dir, f"{base_name}_legend.png")

                self.result_image.save(map_path)
                self.legend_image.save(legend_path)

                QMessageBox.information(self, "Success", f"Results successfully saved to:\n{save_dir}")
            except Exception as e:
                self.show_error(f"Failed to save results: {e}")

    def update_progress(self, message):
        if message.startswith("PROGRESS:"):
            clean_message = message.split(":", 1)[1]
            cursor = self.progress_log.textCursor()

            if self.last_line_is_progress:
                cursor.movePosition(QTextCursor.MoveOperation.End)
                cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
                cursor.removeSelectedText()
            
            self.progress_log.append(clean_message)
            self.last_line_is_progress = True
        else:
            self.progress_log.append(message)
            self.last_line_is_progress = False

        self.progress_log.ensureCursorVisible()

    def analysis_finished(self, result_tuple):
        self.run_btn.setText("Run Analysis")
        self.check_inputs()
        
        result_image, legend_image = result_tuple if result_tuple else (None, None)

        if isinstance(result_image, Image):
            self.update_progress("Backend processing complete.")
            pixmap = result_image.toqpixmap()
            self.image_viewer.set_image(pixmap)
            self.view_stack_layout.setCurrentWidget(self.image_viewer)
            self.result_image = result_image
            
            self.set_border_style(active=False)

            if isinstance(legend_image, Image):
                self.legend_image = legend_image
                legend_pixmap = legend_image.toqpixmap()
                
                scaled_pixmap = legend_pixmap.scaledToHeight(
                    self.stats_scroll_area.height() - 15,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.stats_legend_label.setPixmap(scaled_pixmap)
                self.stats_legend_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
                self.save_as_btn.setEnabled(True)
            else:
                self.stats_legend_label.setText("Could not generate legend.")
                self.stats_legend_label.setAlignment(Qt.AlignCenter)

        else:
            self.placeholder_label.setText("Analysis finished with no result, or an error occurred.")
            self.view_stack_layout.setCurrentWidget(self.placeholder_label)
            self.result_image = None
            self.legend_image = None
            self.save_as_btn.setEnabled(False)
            self.stats_legend_label.setText("Analysis failed.")
            self.stats_legend_label.setAlignment(Qt.AlignCenter)
            self.set_border_style(active=True)

    def show_error(self, error_message):
        self.progress_log.append(f"\nERROR: {error_message}\n")
        QMessageBox.critical(self, "Error", error_message)
        self.analysis_finished(None)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())