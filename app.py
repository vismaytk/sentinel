#!/usr/bin/env python3
"""
SENTINEL - Vehicle Intelligence Platform

Production-grade defence intelligence system for real-time 
vehicle detection, tracking, and license plate recognition.

Usage: python app.py
"""

import logging
import signal
import sys
import atexit
from logging.handlers import RotatingFileHandler
from uuid import uuid4

from config import get_config
from core import (
    get_detector, get_camera, get_database, stop_camera, close_database,
    run_validation, get_alert_engine, get_ocr
)
from api import create_app


def setup_logging(cfg):
    """Configure rotating file + console logging."""
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            cfg.LOG_FILE,
            maxBytes=cfg.LOG_MAX_BYTES,
            backupCount=cfg.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
    ]
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True
    )


def print_banner():
    """Print startup banner with system info."""
    cfg = get_config()
    
    banner = f"""
+==============================================================+
|                                                              |
|   SENTINEL - VEHICLE INTELLIGENCE PLATFORM                   |
|                                                              |
+==============================================================+

  Dashboard:    http://localhost:{cfg.PORT}
  Analytics:    http://localhost:{cfg.PORT}/analytics
  Camera:       {cfg.IP_CAM_URL}
  
  +-----------------------------------------------------------+
  |  FEATURES                                                 |
  |  +-- Detection:   YOLOv8 @ {cfg.YOLO_IMGSZ}px (every {cfg.DETECT_EVERY_N} frames)         |
  |  +-- Tracking:    {'Enabled' if cfg.ENABLE_TRACKING else 'Disabled'}                                |
  |  +-- OCR:         {'Enabled' if cfg.ENABLE_OCR else 'Disabled'}                                 |
  |  +-- TTA:         {'Enabled' if cfg.ENABLE_TTA else 'Disabled'}                                |
  |  +-- Multi-Scale: {'Enabled' if cfg.MULTI_SCALE_INFERENCE else 'Disabled'}                                |
  |  +-- Database:    SQLite (WAL mode)                       |
  +-----------------------------------------------------------+
"""
    print(banner)


def graceful_shutdown(signum=None, frame=None):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    print("\n  [!] Shutting down SENTINEL...")
    
    # Stop camera
    stop_camera()
    
    # Close database (flushes write queue)
    close_database()
    
    print("  [OK] Shutdown complete")
    sys.exit(0)


def main():
    """Main entry point."""
    cfg = get_config()
    
    # Setup logging first
    setup_logging(cfg)
    
    # Print startup banner
    print_banner()
    
    # Run system validation
    print("  [*] Running system validation...")
    if not run_validation():
        print("  [!] Aborting due to validation errors.")
        sys.exit(1)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    atexit.register(graceful_shutdown)
    
    # Initialize database
    print("  [*] Initializing database...")
    db = get_database()
    
    # Run DB integrity check
    integrity = db.verify_integrity()
    if not integrity.get("ok"):
        print(f"  [!] Database integrity warning: {integrity.get('details')}")
    
    # Warmup detection models
    print("  [*] Warming up detection models...")
    detector = get_detector()
    detector.warmup()

    # Warmup OCR model to avoid first plate timeout
    if cfg.ENABLE_OCR:
        print("  [*] Warming up OCR model...")
        get_ocr().warmup()
    
    # Initialize alert engine
    from datetime import datetime
    alert_engine = get_alert_engine()
    alert_engine.set_session_start(datetime.now())
    
    # Start camera stream
    print(f"  [*] Starting camera stream...")
    get_camera(cfg.IP_CAM_URL)
    
    # Create and run Flask app (detection worker starts inside create_app)
    print(f"  [*] Starting web server on port {cfg.PORT}...")
    app = create_app()
    
    # Run with threading enabled for SSE support
    app.run(
        host=cfg.HOST,
        port=cfg.PORT,
        debug=cfg.DEBUG,
        threaded=True,
        use_reloader=False  # Disable reloader to prevent double initialization
    )


if __name__ == "__main__":
    main()
