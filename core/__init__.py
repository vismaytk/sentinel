"""
SENTINEL Core Package

Core modules for the Vehicle Intelligence Platform.
"""

from .database import Database, get_database, close_database
from .detector import Detector, DetectionResult, get_detector
from .tracker import Tracker, Track, get_tracker
from .ocr import PlateOCR, get_ocr
from .camera import CameraStream, get_camera, stop_camera, make_placeholder

__all__ = [
    # Database
    "Database",
    "get_database",
    "close_database",
    
    # Detector
    "Detector",
    "DetectionResult",
    "get_detector",
    
    # Tracker
    "Tracker",
    "Track",
    "get_tracker",
    
    # OCR
    "PlateOCR",
    "get_ocr",
    
    # Camera
    "CameraStream",
    "get_camera",
    "stop_camera",
    "make_placeholder",
]
