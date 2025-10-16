"""
SAM2 Segmentation Annotator - Main Entry Point
"""
import sys
from PyQt6.QtWidgets import QApplication

from main_window import MainWindow
from config import ensure_directories_exist


def main():
    """Main application entry point"""
    # Ensure required directories exist
    ensure_directories_exist()
    
    # Create and run application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
