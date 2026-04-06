"""
SENTINEL Tracker Module

Vehicle tracking using SORT algorithm with fallback to simple IoU-based centroid tracker.
Maintains track IDs across frames for consistent identification.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque

from config import get_config

logger = logging.getLogger(__name__)

# Try to import advanced tracking libraries
try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False

try:
    import lap
    LAP_AVAILABLE = True
except ImportError:
    try:
        from scipy.optimize import linear_sum_assignment
        LAP_AVAILABLE = False
        SCIPY_AVAILABLE = True
    except ImportError:
        LAP_AVAILABLE = False
        SCIPY_AVAILABLE = False


@dataclass
class Track:
    """Single tracked vehicle."""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    last_seen: int  # Frame index
    hits: int = 1  # Number of detections matched
    age: int = 0  # Frames since creation
    is_active: bool = True
    confirmed: bool = False  # Confirmed after TRACK_MIN_HITS frames
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Kalman filter state (if available)
    kf: Optional[object] = None
    
    def __post_init__(self):
        self.bbox_history.append(self.bbox)
        if KALMAN_AVAILABLE:
            self._init_kalman()
    
    def _init_kalman(self):
        """Initialize Kalman filter for position prediction."""
        # State: [x_center, y_center, area, aspect_ratio, vx, vy, va, var]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        
        # Measurement noise
        self.kf.R *= 10
        
        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initial state from bbox
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        area = w * h
        ar = w / max(h, 1)
        
        self.kf.x[:4] = np.array([[cx], [cy], [area], [ar]])
    
    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next position using Kalman filter."""
        if self.kf is None:
            return self.bbox
        
        self.kf.predict()
        
        # Convert state back to bbox
        cx, cy, area, ar = self.kf.x[:4].flatten()
        w = np.sqrt(area * ar)
        h = area / max(w, 1)
        
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        return (x1, y1, x2, y2)
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float, frame_idx: int,
               min_hits_for_confirm: int = 3):
        """Update track with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = frame_idx
        self.hits += 1
        self.bbox_history.append(bbox)
        
        # Confirm track after enough hits
        if self.hits >= min_hits_for_confirm:
            self.confirmed = True
        
        if self.kf is not None:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            area = w * h
            ar = w / max(h, 1)
            
            self.kf.update(np.array([[cx], [cy], [area], [ar]]))
    
    def get_centroid(self) -> Tuple[float, float]:
        """Get center point of bbox."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union between two bboxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = area1 + area2 - inter_area
    
    return inter_area / max(union_area, 1e-6)


def linear_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve linear assignment problem using available library.
    
    Returns:
        Tuple of (row_indices, col_indices) for optimal assignment
    """
    if LAP_AVAILABLE:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.arange(len(x)), x
    elif SCIPY_AVAILABLE:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind
    else:
        # Fallback: greedy assignment
        rows = []
        cols = []
        used_cols = set()
        
        for i in range(cost_matrix.shape[0]):
            best_j = -1
            best_cost = float('inf')
            for j in range(cost_matrix.shape[1]):
                if j not in used_cols and cost_matrix[i, j] < best_cost:
                    best_cost = cost_matrix[i, j]
                    best_j = j
            if best_j >= 0:
                rows.append(i)
                cols.append(best_j)
                used_cols.add(best_j)
        
        return np.array(rows), np.array(cols)


class Tracker:
    """
    Multi-object tracker using SORT algorithm with fallback.
    
    Maintains persistent track IDs across frames using:
    - Kalman filter for motion prediction (if filterpy available)
    - Hungarian algorithm for assignment (if lap/scipy available)
    - IoU-based matching
    """
    
    def __init__(self):
        """Initialize tracker."""
        self.cfg = get_config()
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_idx = 0
        
        # Log available features
        features = []
        if KALMAN_AVAILABLE:
            features.append("Kalman")
        if LAP_AVAILABLE:
            features.append("LAP")
        elif SCIPY_AVAILABLE:
            features.append("SciPy")
        else:
            features.append("Greedy")
        
        logger.info("Tracker initialized: %s", ', '.join(features))
    
    def update(self, detections: List) -> List:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of DetectionResult objects
            
        Returns:
            Same list with track_id field populated
        """
        self.frame_idx += 1
        
        if not self.cfg.ENABLE_TRACKING:
            return detections
        
        # Get detection bboxes and classes
        det_bboxes = [d.bbox for d in detections]
        det_classes = [d.vehicle_type for d in detections]
        
        # Predict track positions
        for track in self.tracks.values():
            track.age += 1
            if track.kf is not None:
                track.bbox = track.predict()
        
        # Match detections to tracks
        if len(det_bboxes) > 0 and len(self.tracks) > 0:
            matched, unmatched_dets, unmatched_tracks = self._match_detections(
                det_bboxes, det_classes, list(self.tracks.values())
            )
            
            # Update matched tracks
            for det_idx, track_id in matched:
                det = detections[det_idx]
                track = self.tracks[track_id]
                track.update(det.bbox, det.confidence, self.frame_idx,
                            min_hits_for_confirm=self.cfg.TRACK_MIN_HITS)
                det.track_id = track_id
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                det = detections[det_idx]
                track = Track(
                    track_id=self.next_id,
                    class_name=det.vehicle_type,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    last_seen=self.frame_idx,
                )
                self.tracks[self.next_id] = track
                det.track_id = self.next_id
                self.next_id += 1
            
            # Mark unmatched tracks
            for track_id in unmatched_tracks:
                track = self.tracks[track_id]
                # Keep track alive for a few frames
                if self.frame_idx - track.last_seen > self.cfg.TRACK_MAX_AGE:
                    track.is_active = False
        
        elif len(det_bboxes) > 0:
            # No existing tracks - create new ones
            for det in detections:
                track = Track(
                    track_id=self.next_id,
                    class_name=det.vehicle_type,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    last_seen=self.frame_idx,
                )
                self.tracks[self.next_id] = track
                det.track_id = self.next_id
                self.next_id += 1
        
        # Remove inactive tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items() 
            if t.is_active and self.frame_idx - t.last_seen <= self.cfg.TRACK_MAX_AGE
        }
        
        return detections
    
    def _match_detections(
        self,
        det_bboxes: List[Tuple[int, int, int, int]],
        det_classes: List[str],
        tracks: List[Track]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using IoU and Hungarian algorithm.
        
        Returns:
            Tuple of (matched pairs, unmatched detection indices, unmatched track IDs)
        """
        if len(det_bboxes) == 0 or len(tracks) == 0:
            return [], list(range(len(det_bboxes))), [t.track_id for t in tracks]
        
        # Compute IoU cost matrix
        cost_matrix = np.zeros((len(det_bboxes), len(tracks)))
        
        for i, det_bbox in enumerate(det_bboxes):
            for j, track in enumerate(tracks):
                # Only match same class
                if det_classes[i] != track.class_name:
                    cost_matrix[i, j] = 1e6  # High cost for class mismatch
                else:
                    iou_score = iou(det_bbox, track.bbox)
                    cost_matrix[i, j] = 1 - iou_score  # Convert to cost
        
        # Solve assignment
        row_indices, col_indices = linear_assignment(cost_matrix)
        
        matched = []
        unmatched_dets = set(range(len(det_bboxes)))
        unmatched_tracks = set(t.track_id for t in tracks)
        
        for i, j in zip(row_indices, col_indices):
            if j < 0 or j >= len(tracks):
                continue
            if cost_matrix[i, j] > 1 - self.cfg.TRACK_IOU_THRESHOLD:
                continue  # IoU too low
            
            track_id = tracks[j].track_id
            matched.append((i, track_id))
            unmatched_dets.discard(i)
            unmatched_tracks.discard(track_id)
        
        return matched, list(unmatched_dets), list(unmatched_tracks)
    
    def get_active_tracks(self) -> List[Dict]:
        """Get list of currently active tracks."""
        return [
            {
                "track_id": t.track_id,
                "class_name": t.class_name,
                "confidence": round(t.confidence, 2),
                "age": t.age,
                "hits": t.hits,
                "bbox": t.bbox,
                "confirmed": t.confirmed,
            }
            for t in self.tracks.values()
            if t.is_active
        ]
    
    def get_confirmed_tracks(self) -> List[Dict]:
        """Get list of confirmed tracks only."""
        return [
            {
                "track_id": t.track_id,
                "class_name": t.class_name,
                "confidence": round(t.confidence, 2),
                "age": t.age,
                "hits": t.hits,
                "bbox": t.bbox,
                "confirmed": t.confirmed,
            }
            for t in self.tracks.values()
            if t.is_active and t.confirmed
        ]
    
    def is_track_confirmed(self, track_id: int) -> bool:
        """Check if a specific track is confirmed."""
        track = self.tracks.get(track_id)
        return track is not None and track.confirmed
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_idx = 0


# Global tracker instance
_tracker: Optional[Tracker] = None


def get_tracker() -> Tracker:
    """Get or create the global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = Tracker()
    return _tracker
