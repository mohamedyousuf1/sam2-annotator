"""
Configuration settings for SAM2 Segmentation Annotator
"""
import os
import torch

# --- Model Configuration ---
# Update these paths to match your environment
MODEL_CHECKPOINT_PATH = r"E:\Segmentation\Annotator\sam2.1_hiera_large.pt"
MODEL_CONFIG = r"E:\Segmentation\Annotator\sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Application Configuration ---
CONFIG_FILE = r"E:\Segmentation\Annotator\annotator_config.txt"  # File to store last used folders

# --- UI Configuration ---
DEFAULT_BRUSH_SIZE = 30  # Default brush/eraser radius in image pixels
ZOOM_IN_FACTOR = 1.15
ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR
MIN_ZOOM = 0.1
MAX_ZOOM = 20.0

# --- Mask Colors (RGBA) ---
BLUE_MASK_COLOR_IMAGE_MODE = (0, 100, 255, 100)
GREEN_MASK_COLOR_IMAGE_MODE = (0, 255, 128, 120)
WHITE_MASK_COLOR_MASK_MODE = (255, 255, 255, 255)

# --- Brush/Eraser Cursor Colors (RGBA) ---
BRUSH_CURSOR_COLOR = (0, 255, 0, 200)
ERASER_CURSOR_COLOR = (255, 0, 0, 200)

# --- Supported Image Formats ---
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp')

# --- Directory Paths ---
ASSETS_DIR = "assets"


def ensure_directories_exist():
    """Ensure required directories exist"""
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
        print(f"Created '{ASSETS_DIR}' directory.")
