"""
Custom QLabel widget for handling image display and mouse interactions
"""
from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, QPoint


class ImageCanvas(QLabel):
    """A custom QLabel to handle mouse events, coordinate scaling, zoom, and pan."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setMouseTracking(True)
        self.set_tool_cursor()  # Set initial cursor based on tool

        # State tracking for interactions
        self.panning = False
        self.is_drawing = False  # For brush/eraser drag
        self.last_pan_pos = QPoint()

    def set_tool_cursor(self):
        """Sets the cursor shape based on the currently selected tool."""
        if not self.parent_window:
            return
            
        tool = self.parent_window.current_tool
        if tool in ['brush', 'eraser']:
            self.setCursor(Qt.CursorShape.BlankCursor)
        elif tool == 'point':
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):
        """Handle mouse wheel scrolling for zooming."""
        if self.parent_window.cv_image is None:
            return

        from config import ZOOM_IN_FACTOR, ZOOM_OUT_FACTOR, MIN_ZOOM, MAX_ZOOM
        
        mouse_pos = event.position()
        point_before_zoom = self.parent_window.transform_point(mouse_pos)

        if event.angleDelta().y() > 0:
            self.parent_window.zoom_level *= ZOOM_IN_FACTOR
        else:
            self.parent_window.zoom_level *= ZOOM_OUT_FACTOR
            
        self.parent_window.zoom_level = max(MIN_ZOOM, min(self.parent_window.zoom_level, MAX_ZOOM))

        if point_before_zoom:
            new_screen_pos = self.parent_window.transform_point_inverse(point_before_zoom)
            self.parent_window.pan_offset += mouse_pos - new_screen_pos
            
        self.parent_window.update_display()

    def mousePressEvent(self, event):
        """Handle mouse press events for different tools."""
        if self.parent_window.cv_image is None:
            return

        # Middle button for panning
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        original_point = self.parent_window.transform_point(event.pos())
        if not original_point:
            return  # Click was outside image bounds

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

    def mouseMoveEvent(self, event):
        """Handle mouse move events for panning and continuous drawing."""
        self.parent_window.mouse_pos = event.pos()  # Track mouse for cursor preview

        if self.panning:
            delta = event.pos() - self.last_pan_pos
            self.parent_window.pan_offset += delta
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

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.MiddleButton and self.panning:
            self.panning = False
            self.set_tool_cursor()

        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False

    def enterEvent(self, event):
        """Handle mouse entering the canvas."""
        self.parent_window.mouse_over_canvas = True
        self.set_tool_cursor()

    def leaveEvent(self, event):
        """Handle mouse leaving the canvas."""
        self.parent_window.mouse_over_canvas = False
        self.parent_window.update_display()  # Redraw to hide cursor preview
