"""
SENTINEL Validation Module

System startup validation to check models, dependencies, camera,
database, and configuration before launching the platform.
"""

import logging
import os
import sys
import sqlite3
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SystemValidator:
    """Validates system requirements before SENTINEL startup."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def check(self, name: str, condition: bool, message: str, level: str = "ERROR") -> bool:
        """
        Record a validation check result.
        
        Args:
            name: Check name
            condition: True if passed
            message: Description or error message
            level: "ERROR" or "WARN"
            
        Returns:
            The condition value
        """
        status = "PASS" if condition else level
        self.results.append({
            "name": name,
            "status": status,
            "message": message
        })
        
        if condition:
            self.passed += 1
        elif level == "ERROR":
            self.failed += 1
        else:
            self.warnings += 1
        
        return condition
    
    def run_all(self) -> bool:
        """
        Run all validation checks.
        
        Returns:
            True if no critical errors
        """
        self._check_models()
        self._check_camera_reachable()
        self._check_dependencies()
        self._check_db_writable()
        self._check_config_values()
        self._print_report()
        return self.failed == 0
    
    def _check_models(self):
        """Verify model files exist and have reasonable size."""
        from config import get_config
        cfg = get_config()
        
        # Vehicle model
        vehicle_path = cfg.VEHICLE_MODEL_PATH
        vehicle_exists = os.path.isfile(vehicle_path)
        self.check(
            "vehicle.pt exists",
            vehicle_exists,
            f"Model not found at {vehicle_path}"
        )
        
        if vehicle_exists:
            size_mb = os.path.getsize(vehicle_path) / 1e6
            self.check(
                "vehicle.pt size",
                size_mb > 1.0,
                f"Model file suspiciously small: {size_mb:.1f}MB"
            )
        
        # Plate model
        plate_path = cfg.PLATE_MODEL_PATH
        plate_exists = os.path.isfile(plate_path)
        self.check(
            "license_plate.pt exists",
            plate_exists,
            f"Model not found at {plate_path}"
        )
        
        if plate_exists:
            size_mb = os.path.getsize(plate_path) / 1e6
            self.check(
                "license_plate.pt size",
                size_mb > 1.0,
                f"Model file suspiciously small: {size_mb:.1f}MB"
            )
        
        # Check vehicle model has expected 4 classes
        if vehicle_exists:
            try:
                from ultralytics import YOLO as _YOLO
                m = _YOLO(vehicle_path)
                expected = {0, 1, 2, 3}
                actual = set(m.names.keys())
                self.check(
                    "vehicle.pt has 4 classes",
                    expected.issubset(actual),
                    f"Expected classes 0-3, got: {actual}",
                    "WARN"
                )
            except Exception:
                pass
    
    def _check_camera_reachable(self):
        """Check if camera source is accessible."""
        from config import get_config
        import cv2
        
        cfg = get_config()
        source = cfg.IP_CAM_URL
        
        # Webcam index
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
            ok = cap.isOpened()
            cap.release()
            self.check(
                "webcam accessible",
                ok,
                f"Webcam {source} not accessible",
                "WARN"
            )
        
        # Local video file
        elif os.path.isfile(source):
            self.check(
                "video file exists",
                True,
                f"Using local file: {source}",
                "WARN"
            )
        
        # IP camera URL
        elif source.startswith("http"):
            try:
                import urllib.request
                urllib.request.urlopen(source, timeout=3)
                self.check("IP camera reachable", True, source)
            except Exception as e:
                self.check(
                    "IP camera reachable",
                    False,
                    f"{source} not reachable: {e}",
                    "WARN"  # WARN because camera may connect later
                )
        
        else:
            self.check(
                "camera source",
                False,
                f"Unknown source type: {source}",
                "WARN"
            )
    
    def _check_dependencies(self):
        """Check required and optional dependencies."""
        deps = {
            "cv2": ("opencv-python", True),
            "ultralytics": ("ultralytics", True),
            "flask": ("flask", True),
            "numpy": ("numpy", True),
            "easyocr": ("easyocr", False),
            "filterpy": ("filterpy", False),
        }
        
        for module, (pkg, required) in deps.items():
            try:
                __import__(module)
                ok = True
            except ImportError:
                ok = False
            
            level = "ERROR" if required else "WARN"
            msg = f"{'Required' if required else 'Optional'} package: pip install {pkg}"
            self.check(f"import {module}", ok, msg, level)
    
    def _check_db_writable(self):
        """Check database directory and write permissions."""
        from config import get_config
        cfg = get_config()
        
        db_dir = os.path.dirname(cfg.DB_PATH)
        
        # Check directory exists
        if db_dir:
            dir_exists = os.path.isdir(db_dir)
            if not dir_exists:
                try:
                    os.makedirs(db_dir, exist_ok=True)
                    dir_exists = True
                except Exception:
                    pass
            
            self.check(
                "data/ directory exists",
                dir_exists,
                f"Create directory: {db_dir}"
            )
        
        # Check write permissions
        try:
            conn = sqlite3.connect(cfg.DB_PATH)
            conn.execute("CREATE TABLE IF NOT EXISTS _test (x INTEGER)")
            conn.execute("DROP TABLE _test")
            conn.close()
            self.check("database writable", True, cfg.DB_PATH)
        except Exception as e:
            self.check("database writable", False, str(e))
    
    def _check_config_values(self):
        """Validate configuration values are within acceptable ranges."""
        from config import get_config
        cfg = get_config()
        
        self.check(
            "JPEG_QUALITY in range",
            10 <= cfg.JPEG_QUALITY <= 95,
            f"JPEG_QUALITY={cfg.JPEG_QUALITY} should be 10-95"
        )
        
        self.check(
            "YOLO_IMGSZ valid",
            cfg.YOLO_IMGSZ in [320, 416, 480, 640],
            f"YOLO_IMGSZ={cfg.YOLO_IMGSZ} should be 320/416/480/640"
        )
        
        self.check(
            "DETECT_EVERY_N valid",
            1 <= cfg.DETECT_EVERY_N <= 10,
            f"DETECT_EVERY_N={cfg.DETECT_EVERY_N} should be 1-10"
        )
        
        self.check(
            "class_conf values",
            all(0.05 <= v <= 0.99 for v in cfg.class_conf.values()),
            f"class_conf values out of range: {cfg.class_conf}"
        )
    
    def _print_report(self):
        """Print validation report to console."""
        print(f"\n  +-- SYSTEM VALIDATION {'--' * 23}+")
        
        for r in self.results:
            if r["status"] == "PASS":
                icon = "[OK]"
            elif r["status"] == "WARN":
                icon = "[!!]"
            else:
                icon = "[XX]"
            
            name = r["name"][:35].ljust(35)
            msg = r["message"][:30]
            print(f"  |  {icon}  {name} {msg}")
        
        print(f"  +{'--' * 29}+")
        print(f"  |  PASSED: {self.passed}  WARNINGS: {self.warnings}  ERRORS: {self.failed}         |")
        print(f"  +{'--' * 29}+\n")
        
        if self.failed > 0:
            print("  [!] Critical errors found. Fix before starting SENTINEL.\n")


def run_validation() -> bool:
    """
    Run all system validation checks.
    
    Returns:
        True if no critical errors
    """
    validator = SystemValidator()
    return validator.run_all()
