import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QStackedLayout
)
from PySide6.QtCore import Qt, QThread, Signal, QRectF
from PySide6.QtGui import QPixmap, QTextCursor, QIcon
from PIL.Image import Image

import backend

class Worker(QThread):
    progress = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, sar_path, optical_path, weights_path):
        super().__init__()
        self.sar_path = sar_path
        self.optical_path = optical_path
        self.weights_path = weights_path

    def run(self):
        try:
            result_image = backend.run_flood_mapping_pipeline(
                sar_tif=self.sar_path,
                optical_tif=self.optical_path,
                weights_file=self.weights_path,
                progress_callback=self.progress.emit,
                output_dir=None
            )
            self.finished.emit(result_image)
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
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom_level = 0

    def wheelEvent(self, event):
        if self._pixmap_item.pixmap().isNull():
            return

        zoom_in, zoom_out = 1.25, 1 / 1.25
        if event.angleDelta().y() > 0:
            factor, self._zoom_level = zoom_in, self._zoom_level + 1
        else:
            factor, self._zoom_level = zoom_out, self._zoom_level - 1

        if self._zoom_level < 0:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
            self._zoom_level = 0
        else:
            self.scale(factor, factor)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flood Mapper")
        self.setGeometry(100, 100, 900, 700)

        self.result_image: Image | None = None

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

        self.save_as_btn = QPushButton("Save Result As...")
        self.save_as_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        self.save_as_btn.clicked.connect(self.save_image_as)
        self.save_as_btn.setEnabled(False)

        self.progress_log = QTextEdit(self)
        self.progress_log.setReadOnly(True)
        self.progress_log.setLineWrapMode(QTextEdit.NoWrap)

        self.image_viewer = ImageViewer()
        self.image_viewer.setMinimumSize(400, 400)

        self.placeholder_label = QLabel("Output will be displayed here.")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("color: gray;")

        self.view_stack_layout = QStackedLayout()
        self.view_stack_layout.addWidget(self.image_viewer)
        self.view_stack_layout.addWidget(self.placeholder_label)

        view_container = QWidget()
        view_container.setLayout(self.view_stack_layout)

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
        left_layout.addWidget(self.progress_log)

        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(view_container, 1)

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
        icon_path = os.path.join(base_path, 'logo.ico')
        self.setWindowIcon(QIcon(icon_path))

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

        self.placeholder_label.setText("Processing, please wait…")
        self.view_stack_layout.setCurrentWidget(self.placeholder_label)
        self.image_viewer.clear_image() 
        
        self.last_line_is_progress = False
        self.result_image = None

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

    def save_image_as(self):
        if not self.result_image:
            QMessageBox.warning(self, "No Image", "There is no analysis result to save.")
            return

        default_name = os.path.splitext(os.path.basename(self.sar_path_edit.text()))[0] + "_flood_map.png"
        
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image As", default_name, "PNG Files (*.png);;All Files (*)")

        if save_path:
            try:
                backend.save_stitched_image(self.result_image, save_path, self.update_progress)
                QMessageBox.information(self, "Success", f"Image successfully saved to:\n{save_path}")
            except Exception as e:
                self.show_error(f"Failed to save image: {e}")

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

    def analysis_finished(self, result_image):
        self.run_btn.setText("Run Analysis")
        self.check_inputs()
        
        if isinstance(result_image, Image):
            self.update_progress("Backend processing complete.")
            pixmap = result_image.toqpixmap()
            self.image_viewer.set_image(pixmap)
            self.view_stack_layout.setCurrentWidget(self.image_viewer)
            self.result_image = result_image
            self.save_as_btn.setEnabled(True)
        else:
            self.placeholder_label.setText("Analysis finished with no result, or an error occurred.")
            self.view_stack_layout.setCurrentWidget(self.placeholder_label)
            self.result_image = None
            self.save_as_btn.setEnabled(False)

    def show_error(self, error_message):
        self.progress_log.append(f"\nERROR: {error_message}\n")
        QMessageBox.critical(self, "Error", error_message)
        self.analysis_finished(None)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())