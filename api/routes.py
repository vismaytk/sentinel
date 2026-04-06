"""
SENTINEL Routes Module

All Flask routes for the Vehicle Intelligence Platform.
"""

import cv2
import logging
import os
import time
import threading
import queue
from collections import deque
from datetime import datetime
from typing import Optional, Generator, Dict, Any, Tuple, List
from uuid import uuid4

from flask import Blueprint, render_template, jsonify, request, Response

from config import get_config, update_config
from core import (
    get_camera, get_detector, get_tracker, get_ocr, get_database,
    make_placeholder, get_alert_engine, get_zone_monitor, Zone,
    get_snapshot_manager
)
from api.events import get_sse_manager

logger = logging.getLogger(__name__)

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
        "alerts": [],
    },
    "fps_history": [],
    "active_tracks": [],
    "auto_detect_n": None,  # Adaptive frame skip
}
_state_lock = threading.Lock()

# Detection queue for async DB writes
_detection_queue: queue.Queue = queue.Queue()

# Required keys for detection entry validation
REQUIRED_DETECTION_KEYS = {"timestamp", "vehicle_type", "confidence", "session_id"}
VALID_VEHICLE_TYPES = {"commercial-vehicle", "military_vehicle"}

# TurboJPEG (optional, faster encoding)
_turbo_jpeg = None
try:
    import turbojpeg
    _turbo_jpeg = turbojpeg.TurboJPEG()
except ImportError:
    pass  # Fall back to cv2.imencode


class DetectionWorker:
    """
    Background worker that decouples detection from MJPEG streaming.
    
    Runs detection pipeline in separate thread, allowing video stream
    to maintain smooth framerate independent of detection speed.
    """
    
    def __init__(self):
        self.output_lock = threading.Lock()
        self.latest_annotated: Optional[Any] = None
        self.latest_detections: List[Any] = []
        self.running = False
        self._thread: Optional[threading.Thread] = None
        
        # Adaptive frame skip tracking
        self._detection_times: deque = deque(maxlen=10)
        self._current_detect_n = get_config().DETECT_EVERY_N
    
    def start(self) -> None:
        """Start the detection worker thread."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the detection worker thread."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
    
    def _adaptive_frame_skip(self, duration_ms: float) -> None:
        """
        Adjust DETECT_EVERY_N based on detection duration.
        
        - If avg duration > 150ms: increase skip (slower detection)
        - If avg duration < 80ms: decrease skip (faster detection)
        """
        self._detection_times.append(duration_ms)
        
        if len(self._detection_times) < 5:
            return
        
        avg_ms = sum(self._detection_times) / len(self._detection_times)
        cfg = get_config()
        
        if avg_ms > 150 and self._current_detect_n < 5:
            self._current_detect_n = min(5, self._current_detect_n + 1)
            update_config(DETECT_EVERY_N=self._current_detect_n)
        elif avg_ms < 80 and self._current_detect_n > 1:
            self._current_detect_n = max(1, self._current_detect_n - 1)
            update_config(DETECT_EVERY_N=self._current_detect_n)
        
        # Update state for API visibility
        with _state_lock:
            _state["auto_detect_n"] = self._current_detect_n
    
    def _loop(self) -> None:
        """Main detection loop running in background thread."""
        cfg = get_config()
        camera = get_camera()
        detector = get_detector()
        tracker = get_tracker()
        ocr = get_ocr()
        zone_monitor = get_zone_monitor()
        snapshot_manager = get_snapshot_manager()
        
        frame_count = 0
        prev_time = time.time()
        
        while self.running:
            frame = camera.read()
            
            if frame is None:
                time.sleep(0.05)
                continue
            
            frame_count += 1
            
            # Only run detection every N frames
            if frame_count % self._current_detect_n != 0:
                time.sleep(0.01)
                continue
            
            # Time the detection
            detect_start = time.time()
            
            # Run detection pipeline
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
            
            # Check zone occupancy and draw zones
            zone_results = None
            if zone_monitor.zones:
                zone_results = zone_monitor.check_detections(
                    detections, frame.shape
                )
                annotated = zone_monitor.draw_zones(annotated, zone_results)
            
            # Auto-capture snapshots
            session_id = get_session_id()
            for det in detections:
                # Check if track is confirmed
                if det.track_id is not None:
                    if not tracker.is_track_confirmed(det.track_id):
                        continue
                
                # Capture on military vehicle
                if det.vehicle_type == "military_vehicle":
                    snapshot_manager.capture(
                        frame, det, "military", session_id
                    )
                
                # Capture on plate read
                if det.plate_text:
                    snapshot_manager.capture(
                        frame, det, "plate", session_id
                    )
            
            detect_end = time.time()
            detect_duration_ms = (detect_end - detect_start) * 1000
            
            # Adaptive frame skip
            self._adaptive_frame_skip(detect_duration_ms)
            
            # Update output buffer
            with self.output_lock:
                self.latest_annotated = annotated
                self.latest_detections = detections
                self._latest_frame = frame  # Store raw frame for manual capture
            
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
            
            time.sleep(0.01)
    
    def get_latest(self) -> Tuple[Optional[Any], List[Any]]:
        """Get the latest annotated frame and detections."""
        with self.output_lock:
            return self.latest_annotated, self.latest_detections
    
    def get_latest_raw_frame(self) -> Optional[Any]:
        """Get the latest raw (unannotated) frame for manual capture."""
        with self.output_lock:
            return getattr(self, '_latest_frame', None)


# Global detection worker
_detection_worker: Optional[DetectionWorker] = None


def get_detection_worker() -> DetectionWorker:
    """Get or create the global detection worker."""
    global _detection_worker
    if _detection_worker is None:
        _detection_worker = DetectionWorker()
    return _detection_worker


def start_detection_worker() -> None:
    """Start the detection worker (call from create_app)."""
    worker = get_detection_worker()
    worker.start()


def validate_detection_entry(det: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate detection entry before DB insertion.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing = REQUIRED_DETECTION_KEYS - set(det.keys())
    if missing:
        return False, f"Missing keys: {missing}"
    
    conf = det.get("confidence", 0)
    if conf < 0 or conf > 1:
        return False, f"confidence out of range: {conf}"
    
    vtype = det.get("vehicle_type")
    if vtype not in VALID_VEHICLE_TYPES:
        return False, f"unknown vehicle_type: {vtype}"
    
    return True, ""


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
    
    # Set session start for alerts
    try:
        alert_engine = get_alert_engine()
        alert_engine.set_session_start(_state["session_start"])
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
        
        # Adaptive frame skip info
        stats["auto_detect_n"] = _state.get("auto_detect_n", cfg.DETECT_EVERY_N)
        
        # Calculate threat level
        mil = stats.get("military_count", 0)
        total = stats.get("vehicles", 0)
        if mil > 5 or (total > 0 and mil / total > 0.3):
            stats["threat_level"] = "HIGH"
        elif mil > 2 or (total > 0 and mil / total > 0.15):
            stats["threat_level"] = "ELEVATED"
        else:
            stats["threat_level"] = "CLEAR"
    
    # Get alerts from alert engine
    try:
        stats["alerts"] = get_alert_engine().get_alerts(10)
    except Exception:
        stats["alerts"] = []
    
    return stats


def update_stats(
    detections: list,
    fps: float,
    camera_status: str,
    active_tracks: list,
    only_confirmed: bool = True
) -> None:
    """
    Update stats from detection results.
    
    Args:
        detections: List of DetectionResult objects
        fps: Current FPS
        camera_status: Camera connection status
        active_tracks: List of active track dicts
        only_confirmed: Only log confirmed tracks to DB
    """
    cfg = get_config()
    tracker = get_tracker()
    
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
            # Check if track is confirmed (if tracking enabled)
            is_confirmed = True
            if det.track_id is not None and only_confirmed:
                is_confirmed = tracker.is_track_confirmed(det.track_id)
            
            entry = {
                "type": det.vehicle_type,
                "conf": round(det.confidence, 2),
                "plate": det.plate_text,
                "track_id": det.track_id,
                "time": datetime.now().strftime("%H:%M:%S"),
                "confirmed": is_confirmed,
            }
            _state["stats"]["detections"].insert(0, entry)
            
            # Update counts (only for confirmed tracks)
            if is_confirmed:
                if det.vehicle_type == "military_vehicle":
                    _state["stats"]["military_count"] += 1
                else:
                    _state["stats"]["commercial_count"] += 1
                
                # Validate and queue for DB write
                db_entry = {
                    "timestamp": det.timestamp,
                    "vehicle_type": det.vehicle_type,
                    "confidence": det.confidence,
                    "track_id": det.track_id,
                    "plate_text": det.plate_text,
                    "plate_conf": det.plate_conf,
                    "bbox": det.bbox,
                    "session_id": _state["session_id"],
                }
                
                is_valid, err = validate_detection_entry(db_entry)
                if is_valid:
                    _detection_queue.put(db_entry)
        
        # Trim detection log
        max_log = cfg.MAX_LOG_ENTRIES
        _state["stats"]["detections"] = _state["stats"]["detections"][:max_log]
        
        # Copy stats for alert evaluation (outside lock)
        stats_snapshot = {
            "vehicles": _state["stats"]["vehicles"],
            "military_count": _state["stats"]["military_count"],
            "commercial_count": _state["stats"]["commercial_count"],
            "plates": _state["stats"]["plates"],
            "fps": fps,
        }
    
    # Evaluate alerts (outside lock to avoid deadlock)
    try:
        get_alert_engine().evaluate(stats_snapshot)
    except Exception:
        pass


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
            logger.error("DB Writer error: %s", e)


def _encode_jpeg(frame, quality: int) -> bytes:
    """
    Encode frame to JPEG using TurboJPEG if available, else cv2.
    
    Args:
        frame: numpy array (BGR)
        quality: JPEG quality (0-100)
        
    Returns:
        JPEG bytes
    """
    global _turbo_jpeg
    
    if _turbo_jpeg is not None:
        try:
            return _turbo_jpeg.encode(frame, quality=quality)
        except Exception:
            pass  # Fall through to cv2
    
    # cv2 fallback
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", frame, encode_params)
    return buffer.tobytes()


def generate_frames() -> Generator[bytes, None, None]:
    """
    MJPEG generator using DetectionWorker for decoupled pipeline.
    
    Detection runs in background thread, streaming pulls latest
    annotated frame without blocking on detection.
    """
    cfg = get_config()
    camera = get_camera()
    worker = get_detection_worker()
    sse = get_sse_manager()
    
    # Ensure SSE has data provider
    sse.set_data_provider(get_stats)
    
    # Placeholder for when no frame is ready
    placeholder = make_placeholder()
    
    while True:
        # Get latest annotated frame from detection worker
        annotated, _ = worker.get_latest()
        
        if annotated is None:
            # Either detection hasn't started or camera not ready
            # Try raw camera frame
            raw_frame = camera.read()
            if raw_frame is not None:
                annotated = raw_frame
            else:
                annotated = placeholder
                time.sleep(0.1)
        
        # Encode and yield
        buffer = _encode_jpeg(annotated, cfg.JPEG_QUALITY)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer + b"\r\n"
        )
        
        # Small sleep to prevent CPU spin
        time.sleep(0.03)  # ~30 FPS max stream rate


def generate_frames_legacy() -> Generator[bytes, None, None]:
    """
    Legacy MJPEG generator (synchronous detection).
    
    Kept for fallback/debugging. Use generate_frames() for production.
    """
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


@bp.route('/report')
def report():
    """Intelligence report page with server-side rendering."""
    from datetime import timedelta
    
    # Get current stats
    stats = get_stats()
    
    # Get analytics data
    db = get_database()
    analytics_data = db.get_analytics(session_id=_state["session_id"])
    
    # Get recent snapshots
    snapshots = get_snapshot_manager().get_snapshots(limit=8)
    
    # Get alerts
    alerts = get_alert_engine().get_alerts(limit=10)
    
    # Calculate duration
    duration = "N/A"
    if _state["session_start"]:
        delta = datetime.now() - _state["session_start"]
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f"{hours}h {minutes}m {seconds}s"
    
    # Process timeline for chart
    timeline = []
    if analytics_data.get("timeline"):
        max_count = max(
            (t.get("commercial", 0) + t.get("military", 0)) 
            for t in analytics_data["timeline"]
        ) or 1
        for t in reversed(analytics_data["timeline"][:30]):  # Last 30 minutes
            total = t.get("commercial", 0) + t.get("military", 0)
            timeline.append({
                "minute": t.get("minute", ""),
                "total": total,
                "height": int((total / max_count) * 100) if max_count > 0 else 0,
            })
    
    # Prepare recent detections
    recent_detections = []
    for det in stats.get("detections", [])[:50]:
        recent_detections.append({
            "time": det.get("time", ""),
            "type": det.get("type", ""),
            "conf": int(det.get("conf", 0) * 100),
            "track_id": det.get("track_id"),
            "plate": det.get("plate"),
        })
    
    return render_template('report.html',
        session_id=stats.get("session_id", ""),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        duration=duration,
        total_vehicles=stats.get("military_count", 0) + stats.get("commercial_count", 0),
        military_count=stats.get("military_count", 0),
        commercial_count=stats.get("commercial_count", 0),
        plates_read=stats.get("plates", 0),
        threat_level=stats.get("threat_level", "CLEAR"),
        timeline=timeline,
        alerts=alerts,
        recent_detections=recent_detections,
        snapshots=snapshots,
        top_plates=analytics_data.get("top_plates", [])[:10],
    )


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
    date_range = request.args.get('range', 'all')  # today | week | all
    
    db = get_database()
    analytics = db.get_analytics(session_id=session_id, date_range=date_range)
    
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
    db = get_database()
    
    with _state_lock:
        fps = _state["stats"].get("fps", 0)
        military = _state["stats"].get("military_count", 0)
        commercial = _state["stats"].get("commercial_count", 0)
    
    return jsonify({
        "status": "ok",
        "camera": camera.status,
        "fps": fps,
        "session_id": _state["session_id"][:8] if _state["session_id"] else None,
        "uptime": (datetime.now() - _state["session_start"]).seconds if _state["session_start"] else 0,
        "detections": {
            "military": military,
            "commercial": commercial,
            "total": military + commercial,
        },
    })


@bp.route('/api/health/db')
def db_health():
    """Database health check endpoint."""
    db = get_database()
    return jsonify(db.verify_integrity())


# ─────────────────────────────────────────────────────────────
# ALERT ROUTES
# ─────────────────────────────────────────────────────────────

@bp.route('/api/alerts')
def api_get_alerts():
    """Get active alerts."""
    limit = request.args.get('limit', 20, type=int)
    alerts = get_alert_engine().get_alerts(limit=limit)
    return jsonify({"alerts": alerts, "count": len(alerts)})


@bp.route('/api/alerts/clear', methods=['POST'])
def api_clear_alerts():
    """Clear all active alerts."""
    get_alert_engine().clear_alerts()
    return jsonify({"ok": True})


@bp.route('/api/alerts/<alert_id>', methods=['DELETE'])
def api_dismiss_alert(alert_id):
    """Dismiss a specific alert."""
    dismissed = get_alert_engine().dismiss_alert(alert_id)
    return jsonify({"ok": dismissed, "alert_id": alert_id})


@bp.route('/api/alerts/rules')
def api_get_alert_rules():
    """Get all alert rules."""
    rules = get_alert_engine().get_rules()
    return jsonify({"rules": rules})


# ─────────────────────────────────────────────────────────────
# ZONE ROUTES
# ─────────────────────────────────────────────────────────────

@bp.route('/api/zones')
def api_get_zones():
    """Get all defined zones."""
    zones = get_zone_monitor().get_zones()
    return jsonify({"zones": zones, "count": len(zones)})


@bp.route('/api/zones', methods=['POST'])
def api_create_zone():
    """Create a new zone."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    
    # Validate required fields
    if 'name' not in data:
        return jsonify({"error": "name is required"}), 422
    if 'polygon' not in data:
        return jsonify({"error": "polygon is required"}), 422
    
    polygon = data.get('polygon', [])
    if len(polygon) < 3:
        return jsonify({"error": "polygon must have at least 3 points"}), 422
    
    # Convert polygon to list of tuples
    try:
        polygon = [tuple(p) for p in polygon]
    except (TypeError, ValueError):
        return jsonify({"error": "invalid polygon format"}), 422
    
    # Create zone
    zone = Zone(
        zone_id=data.get('zone_id', str(uuid4())[:8]),
        name=data['name'],
        polygon=polygon,
        color=tuple(data.get('color', [0, 212, 255])),
        alert_on_military=data.get('alert_on_military', True),
        alert_on_any=data.get('alert_on_any', False),
        enabled=data.get('enabled', True),
    )
    
    get_zone_monitor().add_zone(zone)
    
    return jsonify({"ok": True, "zone_id": zone.zone_id}), 201


@bp.route('/api/zones/events')
def api_zone_events():
    """Get recent zone entry events."""
    limit = request.args.get('limit', 50, type=int)
    events = get_zone_monitor().get_zone_events(limit=limit)
    return jsonify({"events": events, "count": len(events)})


@bp.route('/api/zones/events/clear', methods=['POST'])
def api_clear_zone_events():
    """Clear zone event history."""
    get_zone_monitor().clear_events()
    return jsonify({"ok": True})


@bp.route('/api/zones/occupancy')
def api_zone_occupancy():
    """Get current vehicle count in each zone."""
    occupancy = get_zone_monitor().get_zone_occupancy()
    return jsonify({"occupancy": occupancy})


@bp.route('/api/zones/<zone_id>')
def api_get_zone(zone_id):
    """Get a single zone by ID."""
    zone = get_zone_monitor().get_zone(zone_id)
    if zone is None:
        return jsonify({"error": "Zone not found"}), 404
    return jsonify(zone)


@bp.route('/api/zones/<zone_id>', methods=['PUT'])
def api_update_zone(zone_id):
    """Update a zone."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    
    # Process polygon if provided
    if 'polygon' in data:
        try:
            data['polygon'] = [tuple(p) for p in data['polygon']]
        except (TypeError, ValueError):
            return jsonify({"error": "invalid polygon format"}), 422
    
    # Process color if provided
    if 'color' in data:
        try:
            data['color'] = tuple(data['color'])
        except (TypeError, ValueError):
            return jsonify({"error": "invalid color format"}), 422
    
    updated = get_zone_monitor().update_zone(zone_id, **data)
    
    if not updated:
        return jsonify({"error": "Zone not found"}), 404
    
    return jsonify({"ok": True})


@bp.route('/api/zones/<zone_id>', methods=['DELETE'])
def api_delete_zone(zone_id):
    """Delete a zone."""
    removed = get_zone_monitor().remove_zone(zone_id)
    
    if not removed:
        return jsonify({"error": "Zone not found"}), 404
    
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────
# SNAPSHOT ROUTES
# ─────────────────────────────────────────────────────────────

@bp.route('/api/snapshots')
def api_get_snapshots():
    """Get list of snapshots with optional filtering."""
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    trigger = request.args.get('trigger', None)
    session = request.args.get('session', None)
    
    snapshots = get_snapshot_manager().get_snapshots(
        limit=min(limit, 100),
        offset=offset,
        trigger=trigger,
        session_id=session,
    )
    
    return jsonify({
        "snapshots": snapshots,
        "count": len(snapshots),
        "offset": offset,
        "limit": limit,
    })


@bp.route('/api/snapshots/capture', methods=['POST'])
def api_capture_manual():
    """Manually capture a snapshot of current frame."""
    data = request.get_json() or {}
    note = data.get('note', '')
    
    worker = get_detection_worker()
    frame = worker.get_latest_raw_frame()
    
    if frame is None:
        return jsonify({"error": "No frame available"}), 503
    
    session_id = get_session_id()
    snapshot_id = get_snapshot_manager().capture_manual(frame, session_id, note)
    
    return jsonify({"ok": True, "snapshot_id": snapshot_id}), 201


@bp.route('/api/snapshots/stats')
def api_snapshot_stats():
    """Get snapshot statistics."""
    stats = get_snapshot_manager().get_stats()
    return jsonify(stats)


@bp.route('/api/snapshots/clear', methods=['POST'])
def api_clear_old_snapshots():
    """Clear snapshots older than specified days."""
    data = request.get_json() or {}
    days = data.get('days', 7)
    
    deleted = get_snapshot_manager().clear_old(days=days)
    
    return jsonify({"ok": True, "deleted": deleted})


@bp.route('/api/snapshots/<snapshot_id>')
def api_get_snapshot(snapshot_id):
    """Get a single snapshot by ID."""
    snapshot = get_snapshot_manager().get_snapshot(snapshot_id)
    if snapshot is None:
        return jsonify({"error": "Snapshot not found"}), 404
    return jsonify(snapshot)


@bp.route('/api/snapshots/<snapshot_id>', methods=['DELETE'])
def api_delete_snapshot(snapshot_id):
    """Delete a snapshot."""
    deleted = get_snapshot_manager().delete_snapshot(snapshot_id)
    
    if not deleted:
        return jsonify({"error": "Snapshot not found"}), 404
    
    return jsonify({"ok": True})


@bp.route('/api/snapshots/image/<snapshot_id>')
def api_get_snapshot_image(snapshot_id):
    """Get snapshot image file."""
    from flask import send_file
    
    snapshot = get_snapshot_manager().get_snapshot(snapshot_id)
    if snapshot is None:
        return jsonify({"error": "Snapshot not found"}), 404
    
    image_path = snapshot.get('image_path')
    if not image_path or not os.path.isfile(image_path):
        return jsonify({"error": "Image file not found"}), 404
    
    return send_file(image_path, mimetype='image/jpeg')


@bp.route('/api/snapshots/thumbnail/<snapshot_id>')
def api_get_snapshot_thumbnail(snapshot_id):
    """Get snapshot thumbnail file."""
    from flask import send_file
    
    snapshot = get_snapshot_manager().get_snapshot(snapshot_id)
    if snapshot is None:
        return jsonify({"error": "Snapshot not found"}), 404
    
    thumb_path = snapshot.get('thumbnail_path')
    if not thumb_path or not os.path.isfile(thumb_path):
        return jsonify({"error": "Thumbnail not found"}), 404
    
    return send_file(thumb_path, mimetype='image/jpeg')
