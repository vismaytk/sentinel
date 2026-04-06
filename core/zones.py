"""
SENTINEL Zone Monitoring Module

Track detections inside user-defined polygonal zones.
Triggers alerts when vehicles enter monitored regions.
"""

import cv2
import logging
import numpy as np
import json
import os
import threading
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from uuid import uuid4

from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """Represents a monitored zone in the camera view."""
    zone_id: str
    name: str
    polygon: List[Tuple[int, int]]  # [(x,y), ...] in pixel coords
    color: Tuple[int, int, int] = (0, 212, 255)  # BGR (cyan)
    alert_on_military: bool = True
    alert_on_any: bool = False
    enabled: bool = True


class ZoneMonitor:
    """
    Monitors detections within user-defined polygonal zones.
    
    Features:
    - Load/save zones from config/zones.json
    - Check which detections fall inside each zone
    - Draw zone overlays on frames
    - Track zone entry events
    """
    
    ZONES_PATH = "config/zones.json"
    
    def __init__(self):
        """Initialize zone monitor with zones from config file."""
        self.zones: List[Zone] = []
        self._lock = threading.Lock()
        self._zone_occupancy: Dict[str, List[int]] = {}  # zone_id -> [track_ids]
        self._zone_events: List[Dict] = []  # Recent zone events
        
        # Load zones from config
        self._load_zones()
    
    def _load_zones(self) -> None:
        """Load zones from JSON config file."""
        if not os.path.isfile(self.ZONES_PATH):
            return
        
        try:
            with open(self.ZONES_PATH, 'r') as f:
                data = json.load(f)
            
            for z in data:
                # Convert polygon to list of tuples
                polygon = [tuple(p) for p in z.get('polygon', [])]
                zone = Zone(
                    zone_id=z.get('zone_id', str(uuid4())[:8]),
                    name=z.get('name', 'Unnamed Zone'),
                    polygon=polygon,
                    color=tuple(z.get('color', [0, 212, 255])),
                    alert_on_military=z.get('alert_on_military', True),
                    alert_on_any=z.get('alert_on_any', False),
                    enabled=z.get('enabled', True),
                )
                self.zones.append(zone)
            
            logger.info("Loaded %d zones from %s", len(self.zones), self.ZONES_PATH)
        except Exception as e:
            logger.warning("Error loading zones: %s", e)
    
    def save_zones(self) -> None:
        """Save current zones to JSON config file."""
        os.makedirs(os.path.dirname(self.ZONES_PATH), exist_ok=True)
        
        data = []
        with self._lock:
            for z in self.zones:
                data.append({
                    "zone_id": z.zone_id,
                    "name": z.name,
                    "polygon": [list(p) for p in z.polygon],
                    "color": list(z.color),
                    "alert_on_military": z.alert_on_military,
                    "alert_on_any": z.alert_on_any,
                    "enabled": z.enabled,
                })
        
        with open(self.ZONES_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_zone(self, zone: Zone) -> None:
        """Add a new zone."""
        with self._lock:
            self.zones.append(zone)
        self.save_zones()
    
    def remove_zone(self, zone_id: str) -> bool:
        """Remove a zone by ID. Returns True if removed."""
        with self._lock:
            for i, z in enumerate(self.zones):
                if z.zone_id == zone_id:
                    self.zones.pop(i)
                    self.save_zones()
                    return True
        return False
    
    def update_zone(self, zone_id: str, **kwargs) -> bool:
        """Update zone properties. Returns True if found."""
        with self._lock:
            for z in self.zones:
                if z.zone_id == zone_id:
                    for key, value in kwargs.items():
                        if hasattr(z, key):
                            setattr(z, key, value)
                    self.save_zones()
                    return True
        return False
    
    def get_zones(self) -> List[Dict]:
        """Get all zones as dicts."""
        with self._lock:
            return [asdict(z) for z in self.zones]
    
    def get_zone(self, zone_id: str) -> Optional[Dict]:
        """Get a single zone by ID."""
        with self._lock:
            for z in self.zones:
                if z.zone_id == zone_id:
                    return asdict(z)
        return None
    
    def check_detections(
        self,
        detections: List[Any],
        frame_shape: Tuple[int, int, int]
    ) -> Dict[str, List[Any]]:
        """
        Check which detections fall inside each zone.
        
        Args:
            detections: List of DetectionResult objects
            frame_shape: (height, width, channels) of frame
            
        Returns:
            Dict mapping zone_id to list of detections in that zone
        """
        results: Dict[str, List[Any]] = {}
        new_events = []
        
        with self._lock:
            for zone in self.zones:
                if not zone.enabled or len(zone.polygon) < 3:
                    continue
                
                poly = np.array(zone.polygon, dtype=np.int32)
                zone_dets = []
                current_track_ids = []
                
                for det in detections:
                    # Get center of bounding box
                    cx = (det.bbox[0] + det.bbox[2]) // 2
                    cy = (det.bbox[1] + det.bbox[3]) // 2
                    
                    # Check if center is inside polygon
                    inside = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0
                    
                    if inside:
                        zone_dets.append(det)
                        if det.track_id is not None:
                            current_track_ids.append(det.track_id)
                            
                            # Check for new entry (track wasn't in zone before)
                            prev_tracks = self._zone_occupancy.get(zone.zone_id, [])
                            if det.track_id not in prev_tracks:
                                # New zone entry event
                                event = {
                                    "type": "zone_entry",
                                    "zone_id": zone.zone_id,
                                    "zone_name": zone.name,
                                    "track_id": det.track_id,
                                    "vehicle_type": det.vehicle_type,
                                    "confidence": det.confidence,
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "timestamp": datetime.now().isoformat(),
                                }
                                new_events.append(event)
                
                results[zone.zone_id] = zone_dets
                self._zone_occupancy[zone.zone_id] = current_track_ids
            
            # Store recent events (keep last 100)
            self._zone_events = (new_events + self._zone_events)[:100]
        
        return results
    
    def draw_zones(self, frame: np.ndarray, zone_results: Optional[Dict] = None) -> np.ndarray:
        """
        Draw zone overlays on frame.
        
        Args:
            frame: BGR frame to annotate
            zone_results: Optional dict from check_detections() for highlighting
            
        Returns:
            Annotated frame
        """
        overlay = frame.copy()
        
        with self._lock:
            for zone in self.zones:
                if not zone.enabled or len(zone.polygon) < 3:
                    continue
                
                poly = np.array(zone.polygon, dtype=np.int32)
                color = zone.color
                
                # Check if zone has detections
                has_detections = False
                if zone_results and zone.zone_id in zone_results:
                    has_detections = len(zone_results[zone.zone_id]) > 0
                
                # Fill with transparency
                alpha = 0.3 if has_detections else 0.15
                cv2.fillPoly(overlay, [poly], color)
                
                # Draw border
                border_color = (0, 0, 255) if has_detections else color
                cv2.polylines(frame, [poly], True, border_color, 2)
                
                # Draw zone name
                centroid = poly.mean(axis=0).astype(int)
                cv2.putText(
                    frame, zone.name,
                    (centroid[0] - 30, centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2
                )
                cv2.putText(
                    frame, zone.name,
                    (centroid[0] - 30, centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    border_color, 1
                )
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame
    
    def get_zone_events(self, limit: int = 50) -> List[Dict]:
        """Get recent zone entry events."""
        with self._lock:
            return self._zone_events[:limit]
    
    def get_zone_occupancy(self) -> Dict[str, int]:
        """Get current vehicle count in each zone."""
        with self._lock:
            return {zid: len(tracks) for zid, tracks in self._zone_occupancy.items()}
    
    def clear_events(self) -> None:
        """Clear zone event history."""
        with self._lock:
            self._zone_events.clear()


# Global zone monitor instance
_zone_monitor: Optional[ZoneMonitor] = None


def get_zone_monitor() -> ZoneMonitor:
    """Get or create the global zone monitor instance."""
    global _zone_monitor
    if _zone_monitor is None:
        _zone_monitor = ZoneMonitor()
    return _zone_monitor
