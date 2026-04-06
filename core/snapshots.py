"""
SENTINEL Snapshot Module

Auto-capture and save detection snapshots for intelligence review.
"""

import cv2
import logging
import os
import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from uuid import uuid4

import numpy as np

from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    """Represents a saved detection snapshot."""
    snapshot_id: str
    timestamp: str
    image_path: str
    thumbnail_path: str
    vehicle_type: str
    confidence: float
    track_id: Optional[int]
    plate_text: Optional[str]
    plate_conf: Optional[float]
    bbox: Tuple[int, int, int, int]
    trigger: str  # "military", "plate", "manual", "zone"
    session_id: str
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None


class SnapshotManager:
    """
    Manages automatic and manual snapshot capture.
    
    Features:
    - Auto-capture on military vehicle detection
    - Auto-capture on plate text read
    - Auto-capture on zone entry
    - Manual capture via API
    - Thumbnail generation
    - Metadata storage in JSON
    """
    
    SNAPSHOTS_DIR = "data/snapshots"
    THUMBNAILS_DIR = "data/snapshots/thumbnails"
    INDEX_PATH = "data/snapshots/index.json"
    
    def __init__(self):
        """Initialize snapshot manager."""
        self._lock = threading.Lock()
        self._snapshots: List[Snapshot] = []
        self._track_snapshot_times: Dict[int, float] = {}  # Cooldown per track
        self._cooldown_sec = 5.0  # Min time between snapshots of same track
        
        # Create directories
        os.makedirs(self.SNAPSHOTS_DIR, exist_ok=True)
        os.makedirs(self.THUMBNAILS_DIR, exist_ok=True)
        
        # Load existing snapshots
        self._load_index()
    
    def _load_index(self) -> None:
        """Load snapshot index from disk."""
        if not os.path.isfile(self.INDEX_PATH):
            return
        
        try:
            with open(self.INDEX_PATH, 'r') as f:
                data = json.load(f)
            
            for s in data:
                self._snapshots.append(Snapshot(**s))
            
            logger.info("Loaded %d snapshots from index", len(self._snapshots))
        except Exception as e:
            logger.warning("Error loading snapshot index: %s", e)
    
    def _save_index(self) -> None:
        """Save snapshot index to disk."""
        with self._lock:
            data = [asdict(s) for s in self._snapshots]
        
        with open(self.INDEX_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _should_capture(self, track_id: Optional[int]) -> bool:
        """Check if we should capture (cooldown check)."""
        if track_id is None:
            return True
        
        now = datetime.now().timestamp()
        last_time = self._track_snapshot_times.get(track_id, 0)
        
        if now - last_time < self._cooldown_sec:
            return False
        
        self._track_snapshot_times[track_id] = now
        return True
    
    def capture(
        self,
        frame: np.ndarray,
        detection: Any,
        trigger: str,
        session_id: str,
        zone_id: Optional[str] = None,
        zone_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Capture a snapshot.
        
        Args:
            frame: BGR frame to save
            detection: DetectionResult object
            trigger: What triggered the capture ("military", "plate", "zone", "manual")
            session_id: Current session ID
            zone_id: Optional zone ID if zone-triggered
            zone_name: Optional zone name
            
        Returns:
            Snapshot ID if captured, None if skipped
        """
        # Cooldown check
        if not self._should_capture(detection.track_id):
            return None
        
        snapshot_id = str(uuid4())[:12]
        timestamp = datetime.now()
        
        # File paths
        date_dir = timestamp.strftime("%Y-%m-%d")
        os.makedirs(os.path.join(self.SNAPSHOTS_DIR, date_dir), exist_ok=True)
        
        filename = f"{timestamp.strftime('%H%M%S')}_{snapshot_id}.jpg"
        image_path = os.path.join(self.SNAPSHOTS_DIR, date_dir, filename)
        thumb_filename = f"thumb_{filename}"
        thumbnail_path = os.path.join(self.THUMBNAILS_DIR, thumb_filename)
        
        # Crop around detection with padding
        x1, y1, x2, y2 = detection.bbox
        h, w = frame.shape[:2]
        
        # Add 30% padding
        pad_x = int((x2 - x1) * 0.3)
        pad_y = int((y2 - y1) * 0.3)
        
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)
        
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Save full image
        cv2.imwrite(image_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Generate thumbnail (max 200px wide)
        thumb_h, thumb_w = cropped.shape[:2]
        if thumb_w > 200:
            scale = 200 / thumb_w
            thumb = cv2.resize(cropped, (200, int(thumb_h * scale)))
        else:
            thumb = cropped
        cv2.imwrite(thumbnail_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        # Create snapshot record
        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp.isoformat(),
            image_path=image_path,
            thumbnail_path=thumbnail_path,
            vehicle_type=detection.vehicle_type,
            confidence=detection.confidence,
            track_id=detection.track_id,
            plate_text=detection.plate_text,
            plate_conf=detection.plate_conf,
            bbox=detection.bbox,
            trigger=trigger,
            session_id=session_id,
            zone_id=zone_id,
            zone_name=zone_name,
        )
        
        with self._lock:
            self._snapshots.insert(0, snapshot)
            # Keep only last 1000 snapshots in memory
            self._snapshots = self._snapshots[:1000]
        
        # Save index (async to not block)
        threading.Thread(target=self._save_index, daemon=True).start()
        
        return snapshot_id
    
    def capture_manual(
        self,
        frame: np.ndarray,
        session_id: str,
        note: str = "",
    ) -> str:
        """
        Capture a manual snapshot of entire frame.
        
        Args:
            frame: BGR frame to save
            session_id: Current session ID
            note: Optional note/description
            
        Returns:
            Snapshot ID
        """
        snapshot_id = str(uuid4())[:12]
        timestamp = datetime.now()
        
        # File paths
        date_dir = timestamp.strftime("%Y-%m-%d")
        os.makedirs(os.path.join(self.SNAPSHOTS_DIR, date_dir), exist_ok=True)
        
        filename = f"{timestamp.strftime('%H%M%S')}_manual_{snapshot_id}.jpg"
        image_path = os.path.join(self.SNAPSHOTS_DIR, date_dir, filename)
        thumb_filename = f"thumb_{filename}"
        thumbnail_path = os.path.join(self.THUMBNAILS_DIR, thumb_filename)
        
        # Save full frame
        cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Generate thumbnail
        h, w = frame.shape[:2]
        scale = 200 / w
        thumb = cv2.resize(frame, (200, int(h * scale)))
        cv2.imwrite(thumbnail_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        # Create snapshot record (no detection info)
        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp.isoformat(),
            image_path=image_path,
            thumbnail_path=thumbnail_path,
            vehicle_type="manual",
            confidence=0.0,
            track_id=None,
            plate_text=note if note else None,
            plate_conf=None,
            bbox=(0, 0, w, h),
            trigger="manual",
            session_id=session_id,
        )
        
        with self._lock:
            self._snapshots.insert(0, snapshot)
            self._snapshots = self._snapshots[:1000]
        
        threading.Thread(target=self._save_index, daemon=True).start()
        
        return snapshot_id
    
    def get_snapshots(
        self,
        limit: int = 50,
        offset: int = 0,
        trigger: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict]:
        """Get snapshots with optional filtering."""
        with self._lock:
            filtered = self._snapshots
            
            if trigger:
                filtered = [s for s in filtered if s.trigger == trigger]
            
            if session_id:
                filtered = [s for s in filtered if s.session_id == session_id]
            
            paginated = filtered[offset:offset + limit]
            return [asdict(s) for s in paginated]
    
    def get_snapshot(self, snapshot_id: str) -> Optional[Dict]:
        """Get a single snapshot by ID."""
        with self._lock:
            for s in self._snapshots:
                if s.snapshot_id == snapshot_id:
                    return asdict(s)
        return None
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot and its files."""
        with self._lock:
            for i, s in enumerate(self._snapshots):
                if s.snapshot_id == snapshot_id:
                    # Delete files
                    try:
                        if os.path.isfile(s.image_path):
                            os.remove(s.image_path)
                        if os.path.isfile(s.thumbnail_path):
                            os.remove(s.thumbnail_path)
                    except Exception:
                        pass
                    
                    self._snapshots.pop(i)
                    threading.Thread(target=self._save_index, daemon=True).start()
                    return True
        return False
    
    def get_stats(self) -> Dict:
        """Get snapshot statistics."""
        with self._lock:
            total = len(self._snapshots)
            by_trigger = {}
            for s in self._snapshots:
                by_trigger[s.trigger] = by_trigger.get(s.trigger, 0) + 1
            
            return {
                "total": total,
                "by_trigger": by_trigger,
            }
    
    def clear_old(self, days: int = 7) -> int:
        """Delete snapshots older than specified days."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0
        
        with self._lock:
            remaining = []
            for s in self._snapshots:
                ts = datetime.fromisoformat(s.timestamp)
                if ts < cutoff:
                    # Delete files
                    try:
                        if os.path.isfile(s.image_path):
                            os.remove(s.image_path)
                        if os.path.isfile(s.thumbnail_path):
                            os.remove(s.thumbnail_path)
                    except Exception:
                        pass
                    deleted += 1
                else:
                    remaining.append(s)
            
            self._snapshots = remaining
        
        if deleted > 0:
            self._save_index()
        
        return deleted


# Global snapshot manager instance
_snapshot_manager: Optional[SnapshotManager] = None


def get_snapshot_manager() -> SnapshotManager:
    """Get or create the global snapshot manager instance."""
    global _snapshot_manager
    if _snapshot_manager is None:
        _snapshot_manager = SnapshotManager()
    return _snapshot_manager
