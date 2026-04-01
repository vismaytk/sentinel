"""
SENTINEL Routes Module

All Flask routes for the Vehicle Intelligence Platform.
"""

import cv2
import time
import threading
import queue
from datetime import datetime
from typing import Optional, Generator
from uuid import uuid4

from flask import Blueprint, render_template, jsonify, request, Response

from config import get_config, update_config
from core import (
    get_camera, get_detector, get_tracker, get_ocr, get_database,
    make_placeholder
)
from api.events import get_sse_manager


# Create blueprint
bp = Blueprint('main', __name__)

# Shared state
_state = {
    "session_id": "",
    "session_start": None,
    "stats": {
        "vehicles": 0,
        "plates": 0,
        "fps": 0.0,
        "military_count": 0,
        "commercial_count": 0,
        "detections": [],
        "camera_status": "connecting",
    },
    "fps_history": [],
    "active_tracks": [],
}
_state_lock = threading.Lock()

# Detection queue for async DB writes
_detection_queue: queue.Queue = queue.Queue()


def init_session():
    """Initialize a new session."""
    global _state
    _state["session_id"] = str(uuid4())
    _state["session_start"] = datetime.now()
    _state["stats"]["detections"] = []
    _state["fps_history"] = []
    
    # Record session start in DB
    try:
        db = get_database()
        db.start_session(_state["session_id"])
    except Exception:
        pass


def get_session_id() -> str:
    """Get current session ID."""
    return _state["session_id"]


def get_stats() -> dict:
    """Get current stats snapshot (for SSE)."""
    cfg = get_config()
    with _state_lock:
        stats = _state["stats"].copy()
        stats["session_id"] = _state["session_id"][:8] + "..." if _state["session_id"] else ""
        stats["session_start"] = _state["session_start"].isoformat() if _state["session_start"] else None
        stats["active_tracks"] = _state.get("active_tracks", [])
        
        # Calculate threat level
        mil = stats.get("military_count", 0)
        total = stats.get("vehicles", 0)
        if mil > 5 or (total > 0 and mil / total > 0.3):
            stats["threat_level"] = "HIGH"
        elif mil > 2 or (total > 0 and mil / total > 0.15):
            stats["threat_level"] = "ELEVATED"
        else:
            stats["threat_level"] = "CLEAR"
    
    return stats


def update_stats(
    detections: list,
    fps: float,
    camera_status: str,
    active_tracks: list
) -> None:
    """Update stats from detection results."""
    cfg = get_config()
    
    with _state_lock:
        _state["stats"]["vehicles"] = len(detections)
        _state["stats"]["plates"] = sum(1 for d in detections if d.plate_text)
        _state["stats"]["fps"] = round(fps, 1)
        _state["stats"]["camera_status"] = camera_status
        _state["active_tracks"] = active_tracks
        
        # Track FPS history for sparkline
        _state["fps_history"].append(fps)
        if len(_state["fps_history"]) > 60:
            _state["fps_history"] = _state["fps_history"][-60:]
        
        # Add new detections to log
        for det in detections:
            entry = {
                "type": det.vehicle_type,
                "conf": round(det.confidence, 2),
                "plate": det.plate_text,
                "track_id": det.track_id,
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            _state["stats"]["detections"].insert(0, entry)
            
            # Update counts
            if det.vehicle_type == "military_vehicle":
                _state["stats"]["military_count"] += 1
            else:
                _state["stats"]["commercial_count"] += 1
            
            # Queue for DB write
            _detection_queue.put({
                "timestamp": det.timestamp,
                "vehicle_type": det.vehicle_type,
                "confidence": det.confidence,
                "track_id": det.track_id,
                "plate_text": det.plate_text,
                "plate_conf": det.plate_conf,
                "bbox": det.bbox,
                "session_id": _state["session_id"],
            })
        
        # Trim detection log
        max_log = cfg.MAX_LOG_ENTRIES
        _state["stats"]["detections"] = _state["stats"]["detections"][:max_log]


def db_writer_loop():
    """Background thread to write detections to database."""
    db = get_database()
    
    while True:
        try:
            det = _detection_queue.get(timeout=1.0)
            if det is None:  # Shutdown signal
                break
            db.insert_detection(det)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[DB Writer] Error: {e}")


def generate_frames() -> Generator[bytes, None, None]:
    """MJPEG generator with detection pipeline."""
    cfg = get_config()
    camera = get_camera()
    detector = get_detector()
    tracker = get_tracker()
    ocr = get_ocr()
    sse = get_sse_manager()
    
    # Set SSE data provider
    sse.set_data_provider(get_stats)
    
    prev_time = time.time()
    frame_count = 0
    cached_annotated = None
    cached_detections = []
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY]
    
    while True:
        frame = camera.read()
        
        if frame is None:
            # Show placeholder while camera is connecting
            placeholder = make_placeholder()
            _, buffer = cv2.imencode(".jpg", placeholder, encode_params)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.5)
            continue
        
        frame_count += 1
        
        # Run detection every N frames
        if frame_count % cfg.DETECT_EVERY_N == 1 or cfg.DETECT_EVERY_N == 1:
            annotated, detections = detector.detect(frame)
            
            # Update tracking
            detections = tracker.update(detections)
            
            # Run OCR on detected plates
            if cfg.ENABLE_OCR:
                for det in detections:
                    if det.plate_bbox:
                        text, conf = ocr.read_plate(
                            frame, det.plate_bbox, det.track_id
                        )
                        if text:
                            det.plate_text = text
                            det.plate_conf = conf
            
            cached_annotated = annotated
            cached_detections = detections
            
            # Calculate FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            
            # Update stats
            update_stats(
                detections,
                fps,
                camera.status,
                tracker.get_active_tracks()
            )
        else:
            annotated = cached_annotated if cached_annotated is not None else frame
        
        # Encode and yield frame
        _, buffer = cv2.imencode(".jpg", annotated, encode_params)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@bp.route('/')
def dashboard():
    """Dashboard page with live feed."""
    return render_template('dashboard.html')


@bp.route('/analytics')
def analytics():
    """Analytics page with charts."""
    return render_template('analytics.html')


@bp.route('/video_feed')
def video_feed():
    """MJPEG stream endpoint."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@bp.route('/stream')
def stream():
    """SSE endpoint for real-time stats push."""
    sse = get_sse_manager()
    sse.set_data_provider(get_stats)
    return sse.create_response()


@bp.route('/stats')
def stats():
    """JSON stats endpoint (fallback for SSE)."""
    return jsonify(get_stats())


@bp.route('/config', methods=['GET'])
def get_config_route():
    """Get current configuration."""
    cfg = get_config()
    return jsonify({
        "vehicle_conf": cfg.class_conf,
        "plate_conf": cfg.plate_conf,
        "yolo_imgsz": cfg.YOLO_IMGSZ,
        "detect_every_n": cfg.DETECT_EVERY_N,
        "enable_ocr": cfg.ENABLE_OCR,
        "enable_tracking": cfg.ENABLE_TRACKING,
    })


@bp.route('/config', methods=['POST'])
def set_config():
    """Update configuration live."""
    cfg = get_config()
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    
    errors = []
    
    # Validate and update vehicle confidence
    if "vehicle_conf" in data:
        for cls_name, val in data["vehicle_conf"].items():
            if cls_name in cfg.class_conf:
                try:
                    conf = max(0.05, min(0.99, float(val)))
                    cfg.class_conf[cls_name] = conf
                except (ValueError, TypeError):
                    errors.append(f"Invalid value for {cls_name}")
    
    # Validate and update plate confidence
    if "plate_conf" in data:
        for cls_name, val in data["plate_conf"].items():
            if cls_name in cfg.plate_conf:
                try:
                    conf = max(0.05, min(0.99, float(val)))
                    cfg.plate_conf[cls_name] = conf
                except (ValueError, TypeError):
                    errors.append(f"Invalid value for {cls_name}")
    
    # Validate and update YOLO image size
    if "yolo_imgsz" in data:
        try:
            imgsz = int(data["yolo_imgsz"])
            if imgsz in [320, 416, 480, 640]:
                update_config(YOLO_IMGSZ=imgsz)
            else:
                errors.append("yolo_imgsz must be 320, 416, 480, or 640")
        except (ValueError, TypeError):
            errors.append("Invalid yolo_imgsz value")
    
    # Validate and update detect every N
    if "detect_every_n" in data:
        try:
            n = max(1, int(data["detect_every_n"]))
            update_config(DETECT_EVERY_N=n)
        except (ValueError, TypeError):
            errors.append("Invalid detect_every_n value")
    
    # Toggle OCR
    if "enable_ocr" in data:
        update_config(ENABLE_OCR=bool(data["enable_ocr"]))
    
    # Toggle tracking
    if "enable_tracking" in data:
        update_config(ENABLE_TRACKING=bool(data["enable_tracking"]))
    
    if errors:
        return jsonify({"status": "partial", "errors": errors}), 400
    
    return jsonify({
        "status": "ok",
        "vehicle_conf": cfg.class_conf,
        "plate_conf": cfg.plate_conf,
    })


@bp.route('/api/detections')
def api_detections():
    """Get detection history from database."""
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    vehicle_type = request.args.get('type', None)
    session_id = request.args.get('session', None)
    
    # Cap limit
    limit = min(limit, 1000)
    
    db = get_database()
    detections = db.get_detections(
        limit=limit,
        offset=offset,
        session_id=session_id,
        vehicle_type=vehicle_type
    )
    
    return jsonify({
        "detections": detections,
        "count": len(detections),
        "offset": offset,
        "limit": limit,
    })


@bp.route('/api/analytics')
def api_analytics():
    """Get analytics aggregates from database."""
    session_id = request.args.get('session', None)
    
    db = get_database()
    analytics = db.get_analytics(session_id=session_id)
    
    # Add FPS history from current session
    with _state_lock:
        fps_hist = _state.get("fps_history", [])
        if fps_hist:
            analytics["performance"] = {
                "avg_fps": round(sum(fps_hist) / len(fps_hist), 1),
                "min_fps": round(min(fps_hist), 1),
                "max_fps": round(max(fps_hist), 1),
            }
    
    return jsonify(analytics)


@bp.route('/api/sessions')
def api_sessions():
    """Get list of past sessions."""
    db = get_database()
    sessions = db.get_recent_sessions(limit=20)
    return jsonify({"sessions": sessions})


@bp.route('/api/clear', methods=['POST'])
def api_clear():
    """Clear old data from database."""
    data = request.get_json() or {}
    
    if not data.get('confirm'):
        return jsonify({"error": "Must send confirm=true"}), 400
    
    days = data.get('days', 7)
    
    db = get_database()
    deleted = db.clear_old_data(days=days)
    
    return jsonify({
        "status": "ok",
        "deleted": deleted,
        "days_kept": days,
    })


@bp.route('/health')
def health():
    """Health check endpoint."""
    camera = get_camera()
    
    with _state_lock:
        fps = _state["stats"].get("fps", 0)
    
    return jsonify({
        "status": "ok",
        "camera": camera.status,
        "fps": fps,
        "session_id": _state["session_id"][:8] if _state["session_id"] else None,
    })
