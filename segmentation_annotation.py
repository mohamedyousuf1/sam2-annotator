import sys
import os
import cv2
import numpy as np
import torch
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QMessageBox, QStyle, QButtonGroup)
from PyQt6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QMouseEvent, QResizeEvent, QIcon, QKeyEvent, 
                         QPainterPath, QPolygonF)
from PyQt6.QtCore import Qt, QPoint, QPointF

# Import SAM2 specific components
# Make sure to install the sam2 package or have it in your project directory
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Global Configuration ---
# Update these paths to match your environment
MODEL_CHECKPOINT_PATH = r"E:\Segmentation\Annotator\sam2.1_hiera_large.pt"
MODEL_CONFIG = r"E:\Segmentation\Annotator\sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = r"E:\Segmentation\Annotator\annotator_config.txt" # File to store last used folders

class ImageCanvas(QLabel):
    """A custom QLabel to handle mouse events, coordinate scaling, zoom, and pan."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setMouseTracking(True)
        self.set_tool_cursor() # Set initial cursor based on tool

        # State tracking for interactions
        self.panning = False
        self.is_drawing = False # For brush/eraser drag
        self.last_pan_pos = QPoint()

    def set_tool_cursor(self):
        """Sets the cursor shape based on the currently selected tool."""
        if not self.parent_window: return
        tool = self.parent_window.current_tool
        if tool in ['brush', 'eraser']:
            self.setCursor(Qt.CursorShape.BlankCursor)
        elif tool == 'point':
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):
        """Handle mouse wheel scrolling for zooming."""
        if self.parent_window.cv_image is None: return

        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        mouse_pos = event.position()
        point_before_zoom = self.parent_window.transform_point(mouse_pos)

        if event.angleDelta().y() > 0:
            self.parent_window.zoom_level *= zoom_in_factor
        else:
            self.parent_window.zoom_level *= zoom_out_factor
        self.parent_window.zoom_level = max(0.1, min(self.parent_window.zoom_level, 20))

        if point_before_zoom:
            new_screen_pos = self.parent_window.transform_point_inverse(point_before_zoom)
            self.parent_window.pan_offset += mouse_pos - new_screen_pos
        self.parent_window.update_display()

    def mousePressEvent(self, event: QMouseEvent):
        if self.parent_window.cv_image is None: return

        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        original_point = self.parent_window.transform_point(event.pos())
        if not original_point: return # Click was outside image bounds

        tool = self.parent_window.current_tool
        if event.button() == Qt.MouseButton.LeftButton:
            if tool in ['brush', 'eraser']:
                self.is_drawing = True
                # The first application of the tool also pushes to history.
                if tool == 'brush':
                    self.parent_window.apply_brush_at_point(original_point, push_history=True)
                elif tool == 'eraser':
                    self.parent_window.apply_eraser_at_point(original_point, push_history=True)
            elif tool == 'point':
                self.parent_window.add_prompt(original_point, 1)

        elif event.button() == Qt.MouseButton.RightButton and tool == 'point':
             self.parent_window.add_prompt(original_point, 0)

    def mouseMoveEvent(self, event: QMouseEvent):
        self.parent_window.mouse_pos = event.pos() # Track mouse for cursor preview

        if self.panning:
            delta = event.pos() - self.last_pan_pos
            self.parent_window.pan_offset += QPointF(delta)
            self.last_pan_pos = event.pos()
            self.parent_window.update_display()
            return

        # Continuous drawing/erasing if left button is held down
        if self.is_drawing and (event.buttons() & Qt.MouseButton.LeftButton):
            original_point = self.parent_window.transform_point(event.pos())
            if original_point:
                tool = self.parent_window.current_tool
                if tool == 'brush':
                    # Subsequent applications in the same drag do not create new history states.
                    self.parent_window.apply_brush_at_point(original_point, push_history=False)
                elif tool == 'eraser':
                    self.parent_window.apply_eraser_at_point(original_point, push_history=False)
        else:
            # If not panning or drawing, just update for cursor preview
            self.parent_window.update_display()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton and self.panning:
            self.panning = False
            self.set_tool_cursor()

        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False

    def enterEvent(self, event):
        self.parent_window.mouse_over_canvas = True
        self.set_tool_cursor()

    def leaveEvent(self, event):
        self.parent_window.mouse_over_canvas = False
        self.parent_window.update_display() # Redraw to hide cursor preview


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Segmentation Assistant")
        self.setGeometry(100, 100, 1600, 900)

        # --- Data Members ---
        self.cv_image, self.predictor_sam2 = None, None
        self.prompt_points, self.prompt_labels = [], []
        self.current_mask, self.annotations = None, []
        self.image_files, self.output_folder = [], ""
        self.current_image_index = -1
        self.pixmap = QPixmap()
        self.fit_scale_factor, self.centering_offset = 1.0, QPoint(0, 0)
        self.zoom_level, self.pan_offset = 1.0, QPointF(0, 0)
        self.show_image = True
        self.last_input_folder, self.last_output_folder = "", ""
        self.current_tool = 'point' # can be 'point', 'brush', 'eraser'
        self.brush_size = 30 # Default brush/eraser radius in image pixels
        self.mouse_pos = None # Stores current mouse position for cursor preview
        self.mouse_over_canvas = False
        self.history = [] # For the undo functionality

        # --- GUI Widgets ---
        self.canvas = ImageCanvas(self)
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_load_folder = QPushButton(" Load Folder")
        self.btn_undo = QPushButton(" Undo Last Action")
        self.btn_clear = QPushButton(" Clear Current Points")
        self.btn_reset_mask = QPushButton(" Reset Mask")
        self.btn_delete_image = QPushButton(" Delete Image")
        self.btn_prev = QPushButton(" << Previous")
        self.btn_next = QPushButton(" Next >>")
        self.btn_toggle_view = QPushButton(" Show Mask Only")
        self.info_label = QLabel("Load a folder to begin.")

        # --- Tool Buttons ---
        self.btn_point_tool = QPushButton(" Point Tool")
        self.btn_brush_tool = QPushButton(" Brush Tool")
        self.btn_eraser_tool = QPushButton(" Eraser Tool")
        self.brush_size_label = QLabel(f"Size: {self.brush_size}")
        self.btn_point_tool.setCheckable(True)
        self.btn_brush_tool.setCheckable(True)
        self.btn_eraser_tool.setCheckable(True)
        self.btn_point_tool.setChecked(True)

        self.tool_button_group = QButtonGroup(self)
        self.tool_button_group.addButton(self.btn_point_tool)
        self.tool_button_group.addButton(self.btn_brush_tool)
        self.tool_button_group.addButton(self.btn_eraser_tool)
        self.tool_button_group.setExclusive(True)

        # --- Add Icons ---
        self.btn_load_folder.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.btn_undo.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogBack))
        self.btn_clear.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
        self.btn_reset_mask.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.btn_delete_image.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        self.btn_prev.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        self.btn_next.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward))
        self.btn_toggle_view.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))

        # --- Add Tooltips ---
        self.btn_load_folder.setToolTip("Load Image Folder (Ctrl+O)")
        self.btn_undo.setToolTip("Undo the last annotation action (Ctrl+Z)")
        self.btn_clear.setToolTip("Clear current SAM points and preview mask (C)")
        self.btn_reset_mask.setToolTip("Delete all masks for the current image (R)")
        self.btn_delete_image.setToolTip("Permanently delete the current image and its mask (Delete)")
        self.btn_prev.setToolTip("Save and go to the previous image (A or Left Arrow)")
        self.btn_next.setToolTip("Save and go to the next image (D or Right Arrow)")
        self.btn_toggle_view.setToolTip("Toggle between image and mask-only view (V or T)")
        self.btn_point_tool.setToolTip("Use points to generate masks (P)\nLeft-click: Add positive point\nRight-click: Add negative point")
        self.btn_brush_tool.setToolTip("Manually paint mask areas (B)\nUse [ and ] to change size")
        self.btn_eraser_tool.setToolTip("Manually erase mask areas (E)\nUse [ and ] to change size")

        # --- Layout ---
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        top_controls_layout = QHBoxLayout()
        top_controls_layout.addWidget(self.btn_load_folder)
        top_controls_layout.addWidget(self.btn_prev)
        top_controls_layout.addWidget(self.btn_next)
        
        top_controls_layout.addSpacing(20)
        top_controls_layout.addWidget(self.btn_point_tool)
        top_controls_layout.addWidget(self.btn_brush_tool)
        top_controls_layout.addWidget(self.btn_eraser_tool)
        top_controls_layout.addWidget(self.brush_size_label)
        top_controls_layout.addSpacing(20)

        top_controls_layout.addWidget(self.btn_clear)
        top_controls_layout.addWidget(self.btn_undo)
        top_controls_layout.addWidget(self.btn_reset_mask)
        top_controls_layout.addWidget(self.btn_delete_image)
        top_controls_layout.addWidget(self.btn_toggle_view)
        top_controls_layout.addStretch()
        top_controls_layout.addWidget(self.info_label)

        main_layout.addLayout(top_controls_layout)
        main_layout.addWidget(self.canvas, 1)
        self.setCentralWidget(central_widget)

        # --- Connections ---
        self.btn_load_folder.clicked.connect(self.load_folder)
        self.btn_undo.clicked.connect(self.undo_last_action)
        self.btn_clear.clicked.connect(self.clear_prompts)
        self.btn_reset_mask.clicked.connect(self.reset_mask)
        self.btn_delete_image.clicked.connect(self.delete_current_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_prev.clicked.connect(self.previous_image)
        self.btn_toggle_view.clicked.connect(self.toggle_view)
        self.tool_button_group.buttonClicked.connect(self.on_tool_selected)

        self.update_button_states()
        self.load_config()
        self.init_model()

    def on_tool_selected(self, button):
        if button == self.btn_point_tool:
            self.current_tool = 'point'
        elif button == self.btn_brush_tool:
            self.current_tool = 'brush'
        elif button == self.btn_eraser_tool:
            self.current_tool = 'eraser'
        self.canvas.set_tool_cursor()
        self.update_display()

    def update_button_states(self):
        folder_loaded = bool(self.image_files)
        self.btn_undo.setEnabled(folder_loaded and bool(self.history))
        self.btn_clear.setEnabled(folder_loaded)
        self.btn_reset_mask.setEnabled(folder_loaded)
        self.btn_delete_image.setEnabled(folder_loaded)
        self.btn_prev.setEnabled(folder_loaded)
        self.btn_next.setEnabled(folder_loaded)
        self.btn_toggle_view.setEnabled(folder_loaded)
        self.btn_point_tool.setEnabled(folder_loaded)
        self.btn_brush_tool.setEnabled(folder_loaded)
        self.btn_eraser_tool.setEnabled(folder_loaded)

    def init_model(self):
        print("Loading SAM2 model...")
        if not os.path.exists(MODEL_CHECKPOINT_PATH) or not os.path.exists(MODEL_CONFIG):
            print("❌ Model files not found.")
            self.btn_load_folder.setText("Model Files Not Found")
            self.btn_load_folder.setEnabled(False)
            return
        try:
            sam_model = build_sam2(MODEL_CONFIG)
            sam_model.to(DEVICE)
            checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
            
            state_dict = checkpoint.get('model', checkpoint)
            sam_model.load_state_dict(state_dict)

            self.predictor_sam2 = SAM2ImagePredictor(sam_model)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            self.btn_load_folder.setText("Model Load Failed")
            self.btn_load_folder.setEnabled(False)


    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    lines = [line.strip() for line in f.readlines()]
                    if len(lines) >= 1: self.last_input_folder = lines[0]
                    if len(lines) >= 2: self.last_output_folder = lines[1]
                print(f"Loaded config: Input='{self.last_input_folder}', Output='{self.last_output_folder}'")
            except Exception as e:
                print(f"Could not load config file: {e}")

    def save_config(self, input_folder, output_folder):
        try:
            with open(CONFIG_FILE, 'w') as f:
                f.write(f"{input_folder}\n")
                f.write(f"{output_folder}\n")
            print("Saved folders to config.")
        except Exception as e:
            print(f"Could not save config file: {e}")
    
    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder", self.last_input_folder)
        if not folder_path: return
        
        suggested_output = self.last_output_folder if self.last_output_folder else folder_path
        output_path = QFileDialog.getExistingDirectory(self, "Select Output Folder for Masks", suggested_output)
        if not output_path: return
        
        self.save_config(folder_path, output_path)
        self.last_input_folder, self.last_output_folder = folder_path, output_path

        self.image_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if self.image_files:
            self.output_folder = output_path
            
            start_index = 0
            # Find the first image that doesn't have a corresponding mask
            for i, image_path in enumerate(self.image_files):
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                mask_path = os.path.join(self.output_folder, f"{base_name}.png")
                if not os.path.exists(mask_path):
                    start_index = i
                    break  # Found the first unannotated one, stop searching

            self.current_image_index = start_index
            self.load_current_image()
        else:
            self.info_label.setText("No images found in folder.")
            self.current_image_index = -1
        self.update_button_states()

    def load_current_image(self):
        image_path = self.image_files[self.current_image_index]
        self.cv_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        if self.predictor_sam2:
            self.predictor_sam2.set_image(self.cv_image)
        self.clear_all()

        mask_path = os.path.join(self.output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}.png")
        if os.path.exists(mask_path):
            loaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if loaded_mask is not None: self.annotations.append(loaded_mask > 0)

        h, w, ch = self.cv_image.shape
        self.pixmap = QPixmap.fromImage(QImage(self.cv_image.data, w, h, ch * w, QImage.Format.Format_RGB888))
        self.info_label.setText(f"{self.current_image_index + 1}/{len(self.image_files)}: {os.path.basename(image_path)}")
        self.update_display()
        self.update_button_states()

    def navigate(self, direction):
        if not self.image_files: return
        self.save_current_mask()
        num_images = len(self.image_files)
        self.current_image_index = (self.current_image_index + direction + num_images) % num_images
        self.load_current_image()

    def next_image(self): self.navigate(1)
    def previous_image(self): self.navigate(-1)

    def save_current_mask(self):
        self._commit_current_mask() # Consolidate any active mask before saving
        if not self.annotations: 
            # If no masks, ensure no file exists for this image
            base_name = os.path.splitext(os.path.basename(self.image_files[self.current_image_index]))[0]
            output_path = os.path.join(self.output_folder, f"{base_name}.png")
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"Removed empty mask file: {output_path}")
            return
        
        final_mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
        for mask in self.annotations:
            if mask is not None and mask.any():
                final_mask = np.logical_or(final_mask, mask).astype(np.uint8)
        final_mask *= 255
        
        base_name = os.path.splitext(os.path.basename(self.image_files[self.current_image_index]))[0]
        output_path = os.path.join(self.output_folder, f"{base_name}.png")
        cv2.imwrite(output_path, final_mask)
        print(f"Mask saved to {output_path}")

    def add_prompt(self, point: QPointF, label: int):
        if not self.prompt_points:
            self.push_state_to_history()
        self.prompt_points.append([point.x(), point.y()])
        self.prompt_labels.append(label)
        self.run_prediction()

    def run_prediction(self):
        if not self.prompt_points:
            self.current_mask = None
            self.update_display()
            return
        if self.predictor_sam2.get_image_embedding() is None: return

        masks, _, _ = self.predictor_sam2.predict(
            point_coords=np.array(self.prompt_points),
            point_labels=np.array(self.prompt_labels),
            multimask_output=False)
        self.current_mask = masks[0]
        self.update_display()

    def push_state_to_history(self):
        current_mask_copy = self.current_mask.copy() if self.current_mask is not None else None
        annotations_copy = [ann.copy() for ann in self.annotations]
        self.history.append((annotations_copy, current_mask_copy))

    def undo_last_action(self):
        if self.history:
            self.annotations, self.current_mask = self.history.pop()
            self.prompt_points, self.prompt_labels = [], []
            print("Undo successful.")
            self.update_display()
            self.update_button_states()
        else:
            print("No more actions to undo.")

    def clear_prompts(self):
        self.prompt_points, self.prompt_labels, self.current_mask = [], [], None
        self.update_display()

    def clear_all(self):
        self.annotations, self.pan_offset, self.zoom_level = [], QPointF(0, 0), 1.0
        self.history = []
        self.clear_prompts()
    
    def reset_mask(self):
        if not self.image_files: return
        image_name = os.path.basename(self.image_files[self.current_image_index])
        print(f"Resetting all masks and prompts for {image_name}")
        self.clear_all()
        base_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(self.output_folder, f"{base_name}.png")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Removed saved mask: {output_path}")
            except Exception as e:
                print(f"Could not remove mask file: {e}")
        self.update_display()

    def delete_current_image(self):
        if not self.image_files or self.current_image_index < 0:
            return

        image_path = self.image_files[self.current_image_index]
        image_name = os.path.basename(image_path)
        
        reply = QMessageBox.question(self, 'Confirm Delete',
                                     f"Are you sure you want to permanently delete this image and its mask?\n\n{image_name}",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            # Delete the mask file if it exists
            base_name = os.path.splitext(image_name)[0]
            mask_path = os.path.join(self.output_folder, f"{base_name}.png")
            if os.path.exists(mask_path):
                try:
                    os.remove(mask_path)
                    print(f"Deleted mask file: {mask_path}")
                except Exception as e:
                    print(f"Could not remove mask file: {e}")
            
            # Delete the image file
            try:
                os.remove(image_path)
                print(f"Deleted image file: {image_path}")
            except Exception as e:
                print(f"Could not remove image file: {e}")
                QMessageBox.critical(self, "Error", f"Failed to delete image file:\n{e}")
                return # Stop if we can't delete the main file

            # Remove from the list
            self.image_files.pop(self.current_image_index)

            # Handle UI update and navigation
            if not self.image_files:
                # Last image was deleted, reset everything
                self.clear_all()
                self.canvas.setPixmap(QPixmap()) # Clear canvas
                self.info_label.setText("No images left in folder.")
                self.current_image_index = -1
            else:
                # If we deleted the last item in the list, the index needs to go back one
                if self.current_image_index >= len(self.image_files):
                    self.current_image_index = len(self.image_files) - 1
                # Load the image at the now current index
                self.load_current_image()
                
            self.update_button_states()

    def toggle_view(self):
        self.show_image = not self.show_image
        self.btn_toggle_view.setText("Show Mask Only" if self.show_image else "Show Image & Mask")
        self.update_display()

    def update_display(self):
        if self.pixmap.isNull(): return

        scaled_pixmap = self.pixmap.scaled(self.canvas.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.fit_scale_factor = scaled_pixmap.width() / self.pixmap.width() if self.pixmap.width() > 0 else 1
        self.centering_offset = QPoint((self.canvas.width() - scaled_pixmap.width()) // 2, (self.canvas.height() - scaled_pixmap.height()) // 2)

        final_canvas = QImage(self.canvas.size(), QImage.Format.Format_ARGB32)
        final_canvas.fill(Qt.GlobalColor.darkGray if self.show_image else Qt.GlobalColor.black)
        painter = QPainter(final_canvas)
        
        if self.show_image:
            total_scale = self.fit_scale_factor * self.zoom_level
            final_offset = QPointF(self.centering_offset) + self.pan_offset
            target_w, target_h = int(self.pixmap.width() * total_scale), int(self.pixmap.height() * total_scale)
            painter.drawPixmap(QPointF(final_offset.x(), final_offset.y()), self.pixmap.scaled(target_w, target_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation))

        blue_mask_color = QColor(0, 100, 255, 100) if self.show_image else QColor(255, 255, 255, 255)
        green_mask_color = QColor(0, 255, 128, 120) if self.show_image else QColor(255, 255, 255, 255)

        for mask in self.annotations:
            self.draw_mask(painter, mask, blue_mask_color)
        if self.current_mask is not None:
            self.draw_mask(painter, self.current_mask, green_mask_color)
        
        # Draw brush/eraser cursor preview
        if self.mouse_over_canvas and self.current_tool in ['brush', 'eraser']:
            scaled_radius = self.brush_size * (self.fit_scale_factor * self.zoom_level)
            color = QColor(0, 255, 0, 200) if self.current_tool == 'brush' else QColor(255, 0, 0, 200)
            painter.setPen(QPen(color, 1, Qt.PenStyle.SolidLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if self.mouse_pos:
                painter.drawEllipse(self.mouse_pos, int(scaled_radius), int(scaled_radius))

        painter.end()
        self.canvas.setPixmap(QPixmap.fromImage(final_canvas))

    def draw_mask(self, painter: QPainter, mask: np.ndarray, color: QColor):
        if mask is None: return
        mask_image = QImage(mask.shape[1], mask.shape[0], QImage.Format.Format_ARGB32)
        mask_image.fill(Qt.GlobalColor.transparent)
        mask_painter = QPainter(mask_image)
        mask_painter.setBrush(color)
        mask_painter.setPen(Qt.PenStyle.NoPen)
        
        # The first parameter is the image, the second is the retrieval mode, the third is the approximation method.
        # cv2.RETR_CCOMP creates a two-level hierarchy of contours
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is None: return

        path = QPainterPath()
        # Iterate through top-level contours (outer boundaries)
        idx = 0
        while idx >= 0:
            contour = contours[idx]
            polygon = QPolygonF([QPointF(p[0][0], p[0][1]) for p in contour])
            path.addPolygon(polygon)

            # Check for holes within this contour
            child_idx = hierarchy[0][idx][2]
            while child_idx >= 0:
                hole_contour = contours[child_idx]
                hole_polygon = QPolygonF([QPointF(p[0][0], p[0][1]) for p in hole_contour])
                path.addPolygon(hole_polygon)
                child_idx = hierarchy[0][child_idx][0] # Move to next sibling hole

            idx = hierarchy[0][idx][0] # Move to next top-level contour
            
        path.setFillRule(Qt.FillRule.OddEvenFill)
        mask_painter.drawPath(path)
        
        mask_painter.end()
        
        total_scale = self.fit_scale_factor * self.zoom_level
        final_offset = QPointF(self.centering_offset) + self.pan_offset
        target_w, target_h = int(mask.shape[1] * total_scale), int(mask.shape[0] * total_scale)
        if target_w > 0 and target_h > 0:
            painter.drawImage(QPointF(final_offset.x(), final_offset.y()), mask_image.scaled(target_w, target_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
    def transform_point(self, screen_pos: QPointF) -> QPointF or None:
        total_scale = self.fit_scale_factor * self.zoom_level
        if total_scale == 0: return None
        final_offset = QPointF(self.centering_offset) + self.pan_offset
        image_pos = (QPointF(screen_pos) - final_offset) / total_scale
        w, h = self.pixmap.width(), self.pixmap.height()
        if 0 <= image_pos.x() < w and 0 <= image_pos.y() < h:
            return image_pos
        return None

    def transform_point_inverse(self, image_pos: QPointF) -> QPointF:
        total_scale = self.fit_scale_factor * self.zoom_level
        final_offset = QPointF(self.centering_offset) + self.pan_offset
        return (image_pos * total_scale) + final_offset

    def _commit_current_mask(self):
        """Commits the current_mask to the annotations list and clears associated state."""
        if self.current_mask is not None:
            if np.any(self.current_mask):
                self.annotations.append(self.current_mask.copy())
            self.current_mask = None
            self.prompt_points, self.prompt_labels = [], []

    def apply_brush_at_point(self, center_orig: QPointF, push_history: bool = True):
        """Applies a circular brush to the current_mask."""
        if self.cv_image is None: return
        if push_history:
            self.push_state_to_history()
        
        h, w = self.cv_image.shape[:2]
        layer = np.zeros((h, w), dtype=np.uint8)
        center_tuple = (int(center_orig.x()), int(center_orig.y()))
        cv2.circle(layer, center_tuple, self.brush_size, 255, -1)
        
        brush_mask = layer.astype(bool)
        if self.current_mask is None:
            self.current_mask = np.zeros((h, w), dtype=bool)
        self.current_mask = np.logical_or(self.current_mask, brush_mask)
        
        self.update_display()

    def apply_eraser_at_point(self, center_orig: QPointF, push_history: bool = True):
        """Erases a circular area from the unified mask."""
        if self.cv_image is None: return
        if push_history:
            self.push_state_to_history()

        # Before erasing, commit any temporary green mask to the main annotation list.
        self._commit_current_mask()

        # If there are no masks to erase from, do nothing.
        if not self.annotations:
            return

        h, w = self.cv_image.shape[:2]

        # Create the eraser "hole" mask
        eraser_layer = np.zeros((h, w), dtype=np.uint8)
        center_tuple = (int(center_orig.x()), int(center_orig.y()))
        cv2.circle(eraser_layer, center_tuple, self.brush_size, 255, -1)
        inverted_eraser_mask = ~eraser_layer.astype(bool)
        
        # Unify all existing annotation masks into a single layer
        unified_mask = np.zeros((h, w), dtype=bool)
        for ann_mask in self.annotations:
            unified_mask = np.logical_or(unified_mask, ann_mask)

        # Apply the eraser to the unified mask
        erased_mask = np.logical_and(unified_mask, inverted_eraser_mask)
        
        # Replace the entire annotation list with the single, newly erased mask.
        if np.any(erased_mask):
            self.annotations = [erased_mask]
        else:
            self.annotations = []
        
        self.update_display()

    def change_brush_size(self, delta: int):
        """Changes the brush/eraser size and updates the UI label."""
        self.brush_size = max(1, min(500, self.brush_size + delta))
        self.brush_size_label.setText(f"Size: {self.brush_size}")
        self.update_display()

    def resizeEvent(self, event: QResizeEvent):
        self.update_display()
        super().resizeEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_O and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.load_folder()
            return
            
        if self.current_tool in ['brush', 'eraser']:
            if event.key() == Qt.Key.Key_BracketRight:
                self.change_brush_size(5)
                return
            if event.key() == Qt.Key.Key_BracketLeft:
                self.change_brush_size(-5)
                return

        if not self.image_files: return
        key = event.key()
        modifiers = event.modifiers()

        if key in (Qt.Key.Key_D, Qt.Key.Key_Right): self.next_image()
        elif key in (Qt.Key.Key_A, Qt.Key.Key_Left): self.previous_image()
        elif key in (Qt.Key.Key_V, Qt.Key.Key_T): self.toggle_view()
        elif key == Qt.Key.Key_C: self.clear_prompts()
        elif key == Qt.Key.Key_R: self.reset_mask()
        elif key == Qt.Key.Key_Delete: self.delete_current_image()
        elif key == Qt.Key.Key_Z and modifiers == Qt.KeyboardModifier.ControlModifier: self.undo_last_action()
        elif key == Qt.Key.Key_P:
            self.btn_point_tool.setChecked(True)
            self.on_tool_selected(self.btn_point_tool)
        elif key == Qt.Key.Key_B:
            self.btn_brush_tool.setChecked(True)
            self.on_tool_selected(self.btn_brush_tool)
        elif key == Qt.Key.Key_E:
            self.btn_eraser_tool.setChecked(True)
            self.on_tool_selected(self.btn_eraser_tool)

if __name__ == '__main__':
    # Ensure the assets directory exists
    if not os.path.exists('assets'):
        os.makedirs('assets')
        print("Created 'assets' directory. Please place your logo.jpg inside.")
        
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

