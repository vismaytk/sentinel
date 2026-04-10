"""
SENTINEL Alert Engine

Configurable alert system that monitors detection stats and fires
alerts based on customizable rules with cooldown periods.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    condition: str  # Python expression: "military_count > 3", etc.
    severity: str  # "INFO", "WARNING", "CRITICAL"
    cooldown_sec: float = 60.0  # Don't re-fire within this window
    message: str = ""


class AlertEngine:
    """
    Alert engine that evaluates rules against detection stats.
    
    Default rules monitor:
    - High military vehicle count
    - Military ratio of total traffic
    - No detections when camera is active
    - Low FPS/performance issues
    """
    
    DEFAULT_RULES = [
        AlertRule(
            "high_military_count",
            "military_count > 5",
            "CRITICAL",
            cooldown_sec=60.0,
            message="More than 5 military vehicles detected"
        ),
        AlertRule(
            "military_ratio",
            "military_ratio > 0.4",
            "WARNING",
            cooldown_sec=120.0,
            message="Military vehicles >40% of traffic"
        ),
        AlertRule(
            "no_detections",
            "fps > 5 and vehicles == 0 and uptime > 30",
            "INFO",
            cooldown_sec=300.0,
            message="Camera active but no vehicles detected"
        ),
        AlertRule(
            "high_fps_drop",
            "fps < 3 and fps > 0",
            "WARNING",
            cooldown_sec=30.0,
            message="Detection FPS critically low"
        ),
        AlertRule(
            "plate_read_success",
            "plates > 0 and military_count > 0",
            "INFO",
            cooldown_sec=60.0,
            message="Plate successfully read from military vehicle"
        ),
        AlertRule(
            "gun_detected",
            "gun_count > 0",
            "CRITICAL",
            cooldown_sec=30.0,
            message="Gun detected in frame"
        ),
        AlertRule(
            "grenade_detected",
            "grenade_count > 0",
            "CRITICAL",
            cooldown_sec=30.0,
            message="Grenade detected in frame"
        ),
    ]
    
    def __init__(self, rules: Optional[List[AlertRule]] = None):
        """
        Initialize alert engine.
        
        Args:
            rules: Custom rules or None for defaults
        """
        self.rules = rules or self.DEFAULT_RULES.copy()
        self._last_fired: Dict[str, float] = {}
        self._active_alerts: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._session_start: Optional[datetime] = None
    
    def set_session_start(self, start_time: datetime):
        """Set session start time for uptime calculations."""
        self._session_start = start_time
    
    def evaluate(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate all rules against current stats.
        
        Args:
            stats: Current detection stats dict
            
        Returns:
            List of newly fired alerts
        """
        # Build evaluation context
        total = stats.get("vehicles", 0)
        context = {
            "vehicles": total,
            "military_count": stats.get("military_count", 0),
            "commercial_count": stats.get("commercial_count", 0),
            "gun_count": stats.get("gun_count", 0),
            "grenade_count": stats.get("grenade_count", 0),
            "weapon_count": stats.get("gun_count", 0) + stats.get("grenade_count", 0),
            "military_ratio": stats.get("military_count", 0) / max(total, 1),
            "fps": stats.get("fps", 0),
            "plates": stats.get("plates", 0),
            "uptime": (datetime.now() - self._session_start).seconds if self._session_start else 0,
        }
        
        new_alerts = []
        now = time.time()
        
        for rule in self.rules:
            try:
                # Safe eval with restricted builtins
                fired = eval(rule.condition, {"__builtins__": {}}, context)
            except Exception:
                continue
            
            if fired:
                last = self._last_fired.get(rule.name, 0)
                if now - last >= rule.cooldown_sec:
                    alert = {
                        "id": str(uuid4())[:8],
                        "rule": rule.name,
                        "severity": rule.severity,
                        "message": rule.message,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "timestamp": datetime.now().isoformat(),
                        "context": {k: round(v, 2) if isinstance(v, float) else v 
                                   for k, v in context.items()},
                    }
                    self._last_fired[rule.name] = now
                    
                    with self._lock:
                        self._active_alerts.insert(0, alert)
                        # Keep only last 50 alerts
                        self._active_alerts = self._active_alerts[:50]
                    
                    new_alerts.append(alert)
        
        return new_alerts
    
    def get_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get active alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert dicts
        """
        with self._lock:
            return self._active_alerts[:limit]
    
    def clear_alerts(self):
        """Clear all active alerts."""
        with self._lock:
            self._active_alerts.clear()
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """
        Dismiss a specific alert by ID.
        
        Args:
            alert_id: Alert ID to dismiss
            
        Returns:
            True if alert was found and dismissed
        """
        with self._lock:
            for i, alert in enumerate(self._active_alerts):
                if alert["id"] == alert_id:
                    self._active_alerts.pop(i)
                    return True
        return False
    
    def add_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a rule by name.
        
        Returns:
            True if rule was found and removed
        """
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                self.rules.pop(i)
                return True
        return False
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get all rules as dicts."""
        return [
            {
                "name": r.name,
                "condition": r.condition,
                "severity": r.severity,
                "cooldown_sec": r.cooldown_sec,
                "message": r.message,
            }
            for r in self.rules
        ]


# Global alert engine instance
_alert_engine: Optional[AlertEngine] = None


def get_alert_engine() -> AlertEngine:
    """Get or create the global alert engine instance."""
    global _alert_engine
    if _alert_engine is None:
        _alert_engine = AlertEngine()
    return _alert_engine
