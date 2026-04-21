"""
SENTINEL Configuration Module

Centralized configuration for the Vehicle Intelligence Platform.
All tunable constants are defined here for easy modification.
"""

from dataclasses import dataclass, field
from typing import Dict
import os


@dataclass
class Config:
    """Centralized configuration for SENTINEL."""
    
    # ── Camera Settings ──────────────────────────────────────────
    IP_CAM_URL: str = "http://192.168.31.27:8080/video"
    CAMERA_TIMEOUT_MS: int = 5000000  # FFmpeg timeout in microseconds
    
    # ── Server Settings ──────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    DEBUG: bool = False
    
    # ── Model Settings ────────────────────────────────────────────
    VEHICLE_MODEL_PATH: str = "models/vehicle.pt"
    PLATE_MODEL_PATH: str = "models/license_plate.pt"
    
    # ── Inference Settings ────────────────────────────────────────
    INFERENCE_WIDTH: int = 640
    YOLO_IMGSZ: int = 640
    DETECT_EVERY_N: int = 2
    TARGET_CLASSES: tuple = (0, 1, 2, 3)  # commercial-vehicle, military_vehicle, gun, grenade
    
    # Weapon Detection (off by default)
    ENABLE_GUN_DETECTION: bool = True
    ENABLE_GRENADE_DETECTION: bool = True
    
    # ── NMS Settings (per-class) ──────────────────────────────────
    NMS_IOU_MILITARY: float = 0.7  # Lower = less overlap for close targets
    NMS_IOU_COMMERCIAL: float = 0.7
    NMS_IOU_GUN: float = 0.7
    NMS_IOU_GRENADE: float = 0.7
    NMS_IOU_DEFAULT: float = 0.45
    
    # ── Streaming Settings ────────────────────────────────────────
    JPEG_QUALITY: int = 65
    JPEG_QUALITY_SNAPSHOT: int = 92  # Higher quality for saved snapshots
    FRAME_BUFFER_SIZE: int = 2
    
    # ── Per-Class Confidence Thresholds ───────────────────────────
    # These are mutable and can be updated via /config endpoint
    class_conf: Dict[str, float] = field(default_factory=lambda: {
        "commercial-vehicle": 0.50,
        "military_vehicle": 0.50,
        "gun": 0.60,
        "Grenade": 0.60,
    })
    
    plate_conf: Dict[str, float] = field(default_factory=lambda: {
        "License_Plate": 0.50,
    })
    
    # ── Database Settings ─────────────────────────────────────────
    DB_PATH: str = "data/sentinel.db"
    MAX_LOG_ENTRIES: int = 200
    DB_CLEANUP_DAYS: int = 7
    
    # ── Tracking Settings ─────────────────────────────────────────
    ENABLE_TRACKING: bool = True
    TRACK_MAX_AGE: int = 30  # Frames before a track is dropped
    TRACK_MIN_HITS: int = 3  # Min detections before track is confirmed
    TRACK_IOU_THRESHOLD: float = 0.3
    
    # ── OCR Settings ──────────────────────────────────────────────
    ENABLE_OCR: bool = True
    OCR_GPU: bool = False
    OCR_LANGUAGES: tuple = ('en',)
    OCR_TIMEOUT_MS: int = 2500
    OCR_MIN_TEXT_CONF: float = 0.20
    OCR_MIN_MOVEMENT_PX: int = 10  # Min bbox movement to re-run OCR
    OCR_COOLDOWN_SEC: float = 2.0  # Min time between OCR on same track
    OCR_WORKERS: int = 2  # Thread pool workers for parallel OCR
    
    # ── Accuracy Enhancement Settings ─────────────────────────────
    ENABLE_TTA: bool = False  # Test-Time Augmentation (slower, more accurate)
    MULTI_SCALE_INFERENCE: bool = False  # Run at two scales
    MULTI_SCALE_OFFSET: int = 160  # Second scale = YOLO_IMGSZ - offset
    
    # Platt scaling calibration coefficients per class
    PLATT_A: Dict[str, float] = field(default_factory=lambda: {
        "military_vehicle": 3,
        "commercial-vehicle": 3,
        "gun": 3,
        "Grenade": 3,
    })
    PLATT_B: Dict[str, float] = field(default_factory=lambda: {
        "military_vehicle": -1.5,
        "commercial-vehicle": -1.0,
        "gun": -1.2,
        "Grenade": -1.2,
    })
    
    # ── SSE Settings ──────────────────────────────────────────────
    SSE_PUSH_INTERVAL: float = 0.15  # ~6-7 pushes/second
    SSE_HEARTBEAT_INTERVAL: float = 3.0  # Keep-alive ping
    
    # ── Logging Settings ─────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/sentinel.log"
    LOG_MAX_BYTES: int = 5242880   # 5MB
    LOG_BACKUP_COUNT: int = 3
    
    # ── Colors (BGR for OpenCV) ───────────────────────────────────
    COLOR_COMMERCIAL: tuple = (136, 255, 0)    # #00ff88
    COLOR_MILITARY: tuple = (255, 144, 30)     # #1e90ff
    COLOR_PLATE: tuple = (0, 212, 255)         # #ffd400
    COLOR_TEXT_BG: tuple = (0, 0, 0)           # Black
    COLOR_GUN: tuple = (85, 45, 255)           # #ff2d55
    COLOR_GRENADE: tuple = (26, 159, 255)      # #ff9f1a
    
    def get_class_color(self, class_name: str) -> tuple:
        """Get BGR color for a vehicle class."""
        colors = {
            "commercial-vehicle": self.COLOR_COMMERCIAL,
            "military_vehicle": self.COLOR_MILITARY,
            "gun": self.COLOR_GUN,
            "Grenade": self.COLOR_GRENADE,
        }
        return colors.get(class_name, (0, 255, 0))
    
    def get_active_classes(self) -> tuple:
        """Return active YOLO class indices based on enabled toggles."""
        classes = [0, 1]  # Always detect vehicles
        if self.ENABLE_GUN_DETECTION:
            classes.append(2)
        if self.ENABLE_GRENADE_DETECTION:
            classes.append(3)
        return tuple(classes)
    
    def get_nms_iou(self, class_name: str) -> float:
        """Get NMS IoU threshold for a vehicle class."""
        if class_name == "military_vehicle":
            return self.NMS_IOU_MILITARY
        elif class_name == "commercial-vehicle":
            return self.NMS_IOU_COMMERCIAL
        elif class_name == "gun":
            return self.NMS_IOU_GUN
        elif class_name == "Grenade":
            return self.NMS_IOU_GRENADE
        return self.NMS_IOU_DEFAULT


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global config instance."""
    return config


def update_config(**kwargs) -> None:
    """Update config values at runtime."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
