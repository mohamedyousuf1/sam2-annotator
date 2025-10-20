"""
Main application window for SAM2 Segmentation Annotator
"""
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, QWidget,
                             QFileDialog, QHBoxLayout, QMessageBox, QStyle,
                             QButtonGroup, QLabel)
from PyQt6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QResizeEvent,
                         QPainterPath, QPolygonF, QKeyEvent)
from PyQt6.QtCore import Qt, QPoint, QPointF

from image_canvas import ImageCanvas
from model_loader import ModelLoader
from config import (CONFIG_FILE, DEFAULT_BRUSH_SIZE, SUPPORTED_IMAGE_FORMATS,
                    BLUE_MASK_COLOR_IMAGE_MODE, GREEN_MASK_COLOR_IMAGE_MODE,
                    WHITE_MASK_COLOR_MASK_MODE, BRUSH_CURSOR_COLOR, ERASER_CURSOR_COLOR)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Segmentation Assistant")
        self.setGeometry(100, 100, 1600, 900)

        # --- Data Members ---
        self.cv_image = None
        self.predictor_sam2 = None
        self.prompt_points = []
        self.prompt_labels = []
        self.current_mask = None
        self.annotations = []
        self.image_files = []
        self.output_folder = ""
        self.current_image_index = -1
        self.pixmap = QPixmap()
        self.fit_scale_factor = 1.0
        self.centering_offset = QPoint(0, 0)
        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0)
        self.show_image = True
        self.last_input_folder = ""
        self.last_output_folder = ""
        self.current_tool = 'point'  # can be 'point', 'brush', 'eraser'
        self.brush_size = DEFAULT_BRUSH_SIZE
        self.mouse_pos = None
        self.mouse_over_canvas = False
        self.history = []  # For the undo functionality

        # --- Initialize UI ---
        self._init_ui()
        
        # --- Load saved configuration and model ---
        self.load_config()
        self.init_model()

    def _init_ui(self):
        """Initialize the user interface"""
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
        self.path_label = QLabel("")

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

        # Bottom layout to show current output folder path
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(QLabel("Output:"))
        bottom_layout.addWidget(self.path_label, 1)

        main_layout.addLayout(top_controls_layout)
        main_layout.addWidget(self.canvas, 1)
        main_layout.addLayout(bottom_layout)
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

    def init_model(self):
        """Initialize the SAM2 model"""
        self.predictor_sam2 = ModelLoader.load_sam2_model()
        
        if self.predictor_sam2 is None:
            self.btn_load_folder.setText("Model Files Not Found")
            self.btn_load_folder.setEnabled(False)

    def on_tool_selected(self, button):
        """Handle tool selection changes"""
        if button == self.btn_point_tool:
            self.current_tool = 'point'
        elif button == self.btn_brush_tool:
            self.current_tool = 'brush'
        elif button == self.btn_eraser_tool:
            self.current_tool = 'eraser'
        self.canvas.set_tool_cursor()
        self.update_display()

    def update_button_states(self):
        """Update button enabled/disabled states"""
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

    def load_config(self):
        """Load saved configuration from file"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    lines = [line.strip() for line in f.readlines()]
                    if len(lines) >= 1:
                        self.last_input_folder = lines[0]
                    if len(lines) >= 2:
                        self.last_output_folder = lines[1]
                print(f"Loaded config: Input='{self.last_input_folder}', Output='{self.last_output_folder}'")
            except Exception as e:
                print(f"Could not load config file: {e}")

    def save_config(self, input_folder, output_folder):
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, 'w') as f:
                f.write(f"{input_folder}\n")
                f.write(f"{output_folder}\n")
            print("Saved folders to config.")
        except Exception as e:
            print(f"Could not save config file: {e}")
    
    def load_folder(self):
        """Load image folder and set output folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", self.last_input_folder)
        if not folder_path:
            return
        
        suggested_output = self.last_output_folder if self.last_output_folder else folder_path
        output_path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Masks", suggested_output)
        if not output_path:
            return
        
        # Normalize and persist
        output_path = os.path.normpath(output_path)
        folder_path = os.path.normpath(folder_path)
        print(f"Selected input folder: {folder_path}")
        print(f"Selected output folder: {output_path}")
        self.save_config(folder_path, output_path)
        self.last_input_folder = folder_path
        self.last_output_folder = output_path

        self.image_files = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)
        ]
        
        if self.image_files:
            self.output_folder = output_path
            # Ensure output directory exists
            os.makedirs(self.output_folder, exist_ok=True)
            print(f"Output folder set to: {self.output_folder}")
            self.path_label.setText(self.output_folder)
            
            # Find the first image that doesn't have a corresponding mask
            start_index = 0
            for i, image_path in enumerate(self.image_files):
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                mask_path = os.path.join(self.output_folder, f"{base_name}.png")
                if not os.path.exists(mask_path):
                    start_index = i
                    break

            self.current_image_index = start_index
            self.load_current_image()
        else:
            self.info_label.setText("No images found in folder.")
            self.current_image_index = -1
            
        self.update_button_states()

    def load_current_image(self):
        """Load the current image and its mask if it exists"""
        image_path = self.image_files[self.current_image_index]
        self.cv_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        if self.predictor_sam2:
            self.predictor_sam2.set_image(self.cv_image)
            
        self.clear_all()

        # Load existing mask if present
        mask_path = os.path.join(
            self.output_folder,
            f"{os.path.splitext(os.path.basename(image_path))[0]}.png"
        )
        mask_path = os.path.normpath(mask_path)
        print(f"Looking for existing mask at: {mask_path}")
        if os.path.exists(mask_path):
            loaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if loaded_mask is not None:
                self.annotations.append(loaded_mask > 0)

        h, w, ch = self.cv_image.shape
        self.pixmap = QPixmap.fromImage(
            QImage(self.cv_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        )
        
        self.info_label.setText(
            f"{self.current_image_index + 1}/{len(self.image_files)}: "
            f"{os.path.basename(image_path)}"
        )
        
        self.update_display()
        self.update_button_states()

    def navigate(self, direction):
        """Navigate to next or previous image"""
        if not self.image_files:
            return
            
        self.save_current_mask()
        num_images = len(self.image_files)
        self.current_image_index = (self.current_image_index + direction + num_images) % num_images
        self.load_current_image()

    def next_image(self):
        """Navigate to next image"""
        self.navigate(1)
        
    def previous_image(self):
        """Navigate to previous image"""
        self.navigate(-1)

    def save_current_mask(self):
        """Save the current mask to disk"""
        self._commit_current_mask()
        
        base_name = os.path.splitext(
            os.path.basename(self.image_files[self.current_image_index])
        )[0]
        # Ensure output folder exists before saving
        try:
            os.makedirs(self.output_folder, exist_ok=True)
        except Exception as e:
            print(f"Failed to create output folder '{self.output_folder}': {e}")
        output_path = os.path.join(self.output_folder, f"{base_name}.png")
        output_path = os.path.normpath(output_path)
        
        if not self.annotations:
            # If no masks, ensure no file exists for this image
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"Removed empty mask file: {output_path}")
            return
        
        # Combine all masks
        final_mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
        for mask in self.annotations:
            if mask is not None and mask.any():
                final_mask = np.logical_or(final_mask, mask).astype(np.uint8)
        final_mask *= 255
        print(f"Attempting to save mask: base='{base_name}' to folder='{self.output_folder}' -> '{output_path}'")
        success = cv2.imwrite(output_path, final_mask)
        if success:
            print(f"Mask saved to: {output_path}")
        else:
            print(f"Failed to save mask to: {output_path}")

    def add_prompt(self, point: QPointF, label: int):
        """Add a prompt point for SAM2 prediction"""
        if not self.prompt_points:
            self.push_state_to_history()
            
        self.prompt_points.append([point.x(), point.y()])
        self.prompt_labels.append(label)
        self.run_prediction()

    def run_prediction(self):
        """Run SAM2 prediction with current prompts"""
        if not self.prompt_points:
            self.current_mask = None
            self.update_display()
            return
            
        if self.predictor_sam2.get_image_embedding() is None:
            return

        masks, _, _ = self.predictor_sam2.predict(
            point_coords=np.array(self.prompt_points),
            point_labels=np.array(self.prompt_labels),
            multimask_output=False
        )
        self.current_mask = masks[0]
        self.update_display()

    def push_state_to_history(self):
        """Push current state to history for undo functionality"""
        current_mask_copy = self.current_mask.copy() if self.current_mask is not None else None
        annotations_copy = [ann.copy() for ann in self.annotations]
        self.history.append((annotations_copy, current_mask_copy))

    def undo_last_action(self):
        """Undo the last action"""
        if self.history:
            self.annotations, self.current_mask = self.history.pop()
            self.prompt_points = []
            self.prompt_labels = []
            print("Undo successful.")
            self.update_display()
            self.update_button_states()
        else:
            print("No more actions to undo.")

    def clear_prompts(self):
        """Clear current prompts and preview mask"""
        self.prompt_points = []
        self.prompt_labels = []
        self.current_mask = None
        self.update_display()

    def clear_all(self):
        """Clear all annotations and reset view"""
        self.annotations = []
        self.pan_offset = QPointF(0, 0)
        self.zoom_level = 1.0
        self.history = []
        self.clear_prompts()
    
    def reset_mask(self):
        """Reset all masks for current image"""
        if not self.image_files:
            return

        image_name = os.path.basename(self.image_files[self.current_image_index])
        print(f"Resetting all masks and prompts for {image_name}")

        self.clear_all()

        # Remove saved mask file
        base_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(self.output_folder, f"{base_name}.png")
        output_path = os.path.normpath(output_path)
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Removed saved mask: {output_path}")
            except Exception as e:
                print(f"Could not remove mask file: {e}")

        self.update_display()

    def delete_current_image(self):
        """Delete the current image and its mask"""
        if not self.image_files or self.current_image_index < 0:
            return

        image_path = self.image_files[self.current_image_index]
        image_name = os.path.basename(image_path)
        
        reply = QMessageBox.question(
            self, 'Confirm Delete',
            f"Are you sure you want to permanently delete this image and its mask?\n\n{image_name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Delete the mask file if it exists
            base_name = os.path.splitext(image_name)[0]
            mask_path = os.path.join(self.output_folder, f"{base_name}.png")
            mask_path = os.path.normpath(mask_path)
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
                return

            # Remove from the list
            self.image_files.pop(self.current_image_index)

            # Handle UI update and navigation
            if not self.image_files:
                self.clear_all()
                self.canvas.setPixmap(QPixmap())
                self.info_label.setText("No images left in folder.")
                self.current_image_index = -1
            else:
                if self.current_image_index >= len(self.image_files):
                    self.current_image_index = len(self.image_files) - 1
                self.load_current_image()
                
            self.update_button_states()

    def toggle_view(self):
        """Toggle between showing image+mask and mask-only"""
        self.show_image = not self.show_image
        self.btn_toggle_view.setText("Show Mask Only" if self.show_image else "Show Image & Mask")
        self.update_display()

    def update_display(self):
        """Update the canvas display"""
        if self.pixmap.isNull():
            return

        # Calculate scaling and centering
        scaled_pixmap = self.pixmap.scaled(
            self.canvas.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.fit_scale_factor = scaled_pixmap.width() / self.pixmap.width() if self.pixmap.width() > 0 else 1
        self.centering_offset = QPoint(
            (self.canvas.width() - scaled_pixmap.width()) // 2,
            (self.canvas.height() - scaled_pixmap.height()) // 2
        )

        # Create canvas
        final_canvas = QImage(self.canvas.size(), QImage.Format.Format_ARGB32)
        final_canvas.fill(Qt.GlobalColor.darkGray if self.show_image else Qt.GlobalColor.black)
        painter = QPainter(final_canvas)
        
        # Draw image if in image mode
        if self.show_image:
            total_scale = self.fit_scale_factor * self.zoom_level
            final_offset = QPointF(self.centering_offset) + self.pan_offset
            target_w = int(self.pixmap.width() * total_scale)
            target_h = int(self.pixmap.height() * total_scale)
            painter.drawPixmap(
                QPointF(final_offset.x(), final_offset.y()),
                self.pixmap.scaled(
                    target_w, target_h,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

        # Set mask colors based on view mode
        if self.show_image:
            blue_mask_color = QColor(*BLUE_MASK_COLOR_IMAGE_MODE)
            green_mask_color = QColor(*GREEN_MASK_COLOR_IMAGE_MODE)
        else:
            blue_mask_color = QColor(*WHITE_MASK_COLOR_MASK_MODE)
            green_mask_color = QColor(*WHITE_MASK_COLOR_MASK_MODE)

        # Draw masks
        for mask in self.annotations:
            self.draw_mask(painter, mask, blue_mask_color)
        if self.current_mask is not None:
            self.draw_mask(painter, self.current_mask, green_mask_color)
        
        # Draw brush/eraser cursor preview
        if self.mouse_over_canvas and self.current_tool in ['brush', 'eraser']:
            scaled_radius = self.brush_size * (self.fit_scale_factor * self.zoom_level)
            color = QColor(*BRUSH_CURSOR_COLOR) if self.current_tool == 'brush' else QColor(*ERASER_CURSOR_COLOR)
            painter.setPen(QPen(color, 1, Qt.PenStyle.SolidLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if self.mouse_pos:
                painter.drawEllipse(self.mouse_pos, int(scaled_radius), int(scaled_radius))

        painter.end()
        self.canvas.setPixmap(QPixmap.fromImage(final_canvas))

    def draw_mask(self, painter: QPainter, mask: np.ndarray, color: QColor):
        """Draw a mask on the painter"""
        if mask is None:
            return
            
        mask_image = QImage(mask.shape[1], mask.shape[0], QImage.Format.Format_ARGB32)
        mask_image.fill(Qt.GlobalColor.transparent)
        mask_painter = QPainter(mask_image)
        mask_painter.setBrush(color)
        mask_painter.setPen(Qt.PenStyle.NoPen)
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if hierarchy is None:
            return

        path = QPainterPath()
        
        # Iterate through top-level contours
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
                child_idx = hierarchy[0][child_idx][0]

            idx = hierarchy[0][idx][0]
            
        path.setFillRule(Qt.FillRule.OddEvenFill)
        mask_painter.drawPath(path)
        mask_painter.end()
        
        # Scale and draw
        total_scale = self.fit_scale_factor * self.zoom_level
        final_offset = QPointF(self.centering_offset) + self.pan_offset
        target_w = int(mask.shape[1] * total_scale)
        target_h = int(mask.shape[0] * total_scale)
        
        if target_w > 0 and target_h > 0:
            painter.drawImage(
                QPointF(final_offset.x(), final_offset.y()),
                mask_image.scaled(
                    target_w, target_h,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
        
    def transform_point(self, screen_pos):
        """Transform screen coordinates to image coordinates"""
        total_scale = self.fit_scale_factor * self.zoom_level
        if total_scale == 0:
            return None
            
        final_offset = QPointF(self.centering_offset) + self.pan_offset
        image_pos = (QPointF(screen_pos) - final_offset) / total_scale
        
        w, h = self.pixmap.width(), self.pixmap.height()
        if 0 <= image_pos.x() < w and 0 <= image_pos.y() < h:
            return image_pos
        return None

    def transform_point_inverse(self, image_pos: QPointF):
        """Transform image coordinates to screen coordinates"""
        total_scale = self.fit_scale_factor * self.zoom_level
        final_offset = QPointF(self.centering_offset) + self.pan_offset
        return (image_pos * total_scale) + final_offset

    def _commit_current_mask(self):
        """Commits the current_mask to the annotations list and clears associated state"""
        if self.current_mask is not None:
            if np.any(self.current_mask):
                self.annotations.append(self.current_mask.copy())
            self.current_mask = None
            self.prompt_points = []
            self.prompt_labels = []

    def apply_brush_at_point(self, center_orig: QPointF, push_history: bool = True):
        """Apply a circular brush to the current_mask"""
        if self.cv_image is None:
            return
            
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
        """Erase a circular area from the unified mask"""
        if self.cv_image is None:
            return
            
        if push_history:
            self.push_state_to_history()

        # Commit any temporary green mask to the main annotation list
        self._commit_current_mask()

        if not self.annotations:
            return

        h, w = self.cv_image.shape[:2]

        # Create the eraser "hole" mask
        eraser_layer = np.zeros((h, w), dtype=np.uint8)
        center_tuple = (int(center_orig.x()), int(center_orig.y()))
        cv2.circle(eraser_layer, center_tuple, self.brush_size, 255, -1)
        inverted_eraser_mask = ~eraser_layer.astype(bool)
        
        # Unify all existing annotation masks
        unified_mask = np.zeros((h, w), dtype=bool)
        for ann_mask in self.annotations:
            unified_mask = np.logical_or(unified_mask, ann_mask)

        # Apply the eraser
        erased_mask = np.logical_and(unified_mask, inverted_eraser_mask)
        
        # Replace with the single erased mask
        if np.any(erased_mask):
            self.annotations = [erased_mask]
        else:
            self.annotations = []
        
        self.update_display()

    def change_brush_size(self, delta: int):
        """Change the brush/eraser size"""
        self.brush_size = max(1, min(500, self.brush_size + delta))
        self.brush_size_label.setText(f"Size: {self.brush_size}")
        self.update_display()

    def resizeEvent(self, event: QResizeEvent):
        """Handle window resize"""
        self.update_display()
        super().resizeEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts"""
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

        if not self.image_files:
            return
            
        key = event.key()
        modifiers = event.modifiers()

        if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.next_image()
        elif key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.previous_image()
        elif key in (Qt.Key.Key_V, Qt.Key.Key_T):
            self.toggle_view()
        elif key == Qt.Key.Key_C:
            self.clear_prompts()
        elif key == Qt.Key.Key_R:
            self.reset_mask()
        elif key == Qt.Key.Key_Delete:
            self.delete_current_image()
        elif key == Qt.Key.Key_Z and modifiers == Qt.KeyboardModifier.ControlModifier:
            self.undo_last_action()
        elif key == Qt.Key.Key_P:
            self.btn_point_tool.setChecked(True)
            self.on_tool_selected(self.btn_point_tool)
        elif key == Qt.Key.Key_B:
            self.btn_brush_tool.setChecked(True)
            self.on_tool_selected(self.btn_brush_tool)
        elif key == Qt.Key.Key_E:
            self.btn_eraser_tool.setChecked(True)
            self.on_tool_selected(self.btn_eraser_tool)
