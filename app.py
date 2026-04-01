#!/usr/bin/env python3
"""
SENTINEL - Vehicle Intelligence Platform

Production-grade defence intelligence system for real-time 
vehicle detection, tracking, and license plate recognition.

Usage: python app.py
"""

import signal
import sys
import atexit
from uuid import uuid4

from config import get_config
from core import get_detector, get_camera, get_database, stop_camera, close_database
from api import create_app


def print_banner():
    """Print startup banner with system info."""
    cfg = get_config()
    
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗║
║   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║║
║   ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║║
║   ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║║
║   ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███║
║   ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══╝
║                                                              ║
║              VEHICLE INTELLIGENCE PLATFORM                   ║
╚══════════════════════════════════════════════════════════════╝

  Dashboard:    http://localhost:{cfg.PORT}
  Analytics:    http://localhost:{cfg.PORT}/analytics
  Camera:       {cfg.IP_CAM_URL}
  
  ┌─────────────────────────────────────────────────────────────┐
  │  FEATURES                                                   │
  │  ├─ Detection:   YOLOv8 @ {cfg.YOLO_IMGSZ}px (every {cfg.DETECT_EVERY_N} frames)          │
  │  ├─ Tracking:    {'Enabled' if cfg.ENABLE_TRACKING else 'Disabled'}                                         │
  │  ├─ OCR:         {'Enabled' if cfg.ENABLE_OCR else 'Disabled'}                                         │
  │  └─ Database:    SQLite (WAL mode)                          │
  └─────────────────────────────────────────────────────────────┘
"""
    print(banner)


def graceful_shutdown(signum=None, frame=None):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    print("\n  🛑 Shutting down SENTINEL...")
    
    # Stop camera
    stop_camera()
    
    # Close database (flushes write queue)
    close_database()
    
    print("  ✅ Shutdown complete")
    sys.exit(0)


def main():
    """Main entry point."""
    cfg = get_config()
    
    # Print startup banner
    print_banner()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    atexit.register(graceful_shutdown)
    
    # Initialize database
    print("  📊 Initializing database...")
    get_database()
    
    # Warmup detection models
    print("  🔥 Warming up detection models...")
    detector = get_detector()
    detector.warmup()
    
    # Start camera stream
    print(f"  📡 Starting camera stream...")
    get_camera(cfg.IP_CAM_URL)
    
    # Create and run Flask app
    print(f"  🚀 Starting web server on port {cfg.PORT}...")
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
