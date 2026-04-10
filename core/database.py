"""
SENTINEL Database Module

SQLite persistence layer with WAL mode, connection pooling, 
and background write queue for non-blocking operations.
"""

import logging
import sqlite3
import threading
import queue
import atexit
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from config import get_config

logger = logging.getLogger(__name__)


class Database:
    """SQLite database manager with connection pooling and background writes."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database with connection pool."""
        self.db_path = db_path or get_config().DB_PATH
        self._local = threading.local()
        self._write_queue: queue.Queue = queue.Queue()
        self._shutdown = threading.Event()
        
        # Initialize database schema
        self._init_schema()
        
        # Start background writer thread (not daemon - needs to flush on shutdown)
        self._writer_thread = threading.Thread(target=self._write_worker, daemon=False)
        self._writer_thread.start()
        
        # Register cleanup on exit
        atexit.register(self.close)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            self._local.connection = conn
        return self._local.connection
    
    @contextmanager
    def _cursor(self):
        """Context manager for database cursor."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def _init_schema(self) -> None:
        """Create database tables and indexes."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        
        # Main detections table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                track_id INTEGER,
                plate_text TEXT,
                plate_conf REAL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                session_id TEXT NOT NULL
            )
        """)
        
        # Indexes for fast queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_type ON detections(vehicle_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON detections(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_text ON detections(plate_text)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_track_id ON detections(track_id)")
        
        # Sessions table for tracking sessions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_detections INTEGER DEFAULT 0,
                military_count INTEGER DEFAULT 0,
                commercial_count INTEGER DEFAULT 0,
                plate_read_count INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _write_worker(self) -> None:
        """Background thread that processes write queue with batch optimization."""
        import time as _time
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        
        # Batch configuration
        BATCH_MAX_SIZE = 10
        BATCH_MAX_WAIT_MS = 100  # 100ms
        
        INSERT_SQL = """
            INSERT INTO detections 
            (timestamp, vehicle_type, confidence, track_id, plate_text, plate_conf,
             bbox_x1, bbox_y1, bbox_x2, bbox_y2, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        while not self._shutdown.is_set() or not self._write_queue.empty():
            batch = []
            session_queries = []  # Non-detection queries (session start/end)
            deadline = _time.time() + (BATCH_MAX_WAIT_MS / 1000.0)
            
            # Collect batch
            while _time.time() < deadline and len(batch) < BATCH_MAX_SIZE:
                try:
                    remaining = max(0.01, deadline - _time.time())
                    item = self._write_queue.get(timeout=remaining)
                    
                    if item is None:  # Shutdown signal
                        break
                    
                    query, params = item
                    
                    # Separate detection inserts from other queries
                    if "INSERT INTO detections" in query:
                        batch.append(params)
                    else:
                        session_queries.append((query, params))
                    
                    self._write_queue.task_done()
                    
                except queue.Empty:
                    break
            
            # Execute batch insert if we have detection items
            if batch:
                try:
                    conn.executemany(INSERT_SQL, batch)
                    conn.commit()
                except sqlite3.Error as e:
                    logger.error("DB batch write error: %s", e)
            
            # Execute other queries individually
            for query, params in session_queries:
                try:
                    conn.execute(query, params)
                    conn.commit()
                except sqlite3.Error as e:
                    logger.error("DB write error: %s", e)
        
        conn.close()
    
    def insert_detection(self, det: Dict[str, Any]) -> None:
        """
        Queue a detection for insertion (non-blocking).
        
        Args:
            det: Detection dict with keys:
                - timestamp, vehicle_type, confidence, track_id,
                - plate_text, plate_conf, bbox (x1,y1,x2,y2), session_id
        """
        query = """
            INSERT INTO detections 
            (timestamp, vehicle_type, confidence, track_id, plate_text, plate_conf,
             bbox_x1, bbox_y1, bbox_x2, bbox_y2, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        bbox = det.get('bbox', (0, 0, 0, 0))
        params = (
            det.get('timestamp', datetime.now().isoformat()),
            det.get('vehicle_type', 'unknown'),
            det.get('confidence', 0.0),
            det.get('track_id'),
            det.get('plate_text'),
            det.get('plate_conf'),
            bbox[0] if len(bbox) > 0 else 0,
            bbox[1] if len(bbox) > 1 else 0,
            bbox[2] if len(bbox) > 2 else 0,
            bbox[3] if len(bbox) > 3 else 0,
            det.get('session_id', 'unknown')
        )
        self._write_queue.put((query, params))
    
    def get_detections(
        self,
        limit: int = 500,
        offset: int = 0,
        session_id: Optional[str] = None,
        vehicle_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve detections with optional filtering.
        
        Args:
            limit: Maximum number of results
            offset: Pagination offset
            session_id: Filter by session
            vehicle_type: Filter by vehicle type
            
        Returns:
            List of detection dicts
        """
        query = "SELECT * FROM detections WHERE 1=1"
        params = []
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        if vehicle_type:
            query += " AND vehicle_type = ?"
            params.append(vehicle_type)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self._cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_analytics(self, session_id: Optional[str] = None, date_range: str = 'all') -> Dict[str, Any]:
        """
        Get comprehensive analytics data.
        
        Args:
            session_id: Optional session filter
            date_range: 'today' | 'week' | 'all'
        
        Returns dict with:
            - timeline: detections per minute
            - type_distribution: commercial vs military counts
            - confidence_histogram: counts per confidence bucket
            - plate_detection_rate: % with plate text
            - hourly_heatmap: activity by day/hour
            - top_plates: most frequent plate texts
            - performance: FPS stats (placeholder)
        """
        with self._cursor() as cursor:
            # Build date filter
            date_filter = ""
            date_params = []
            if date_range == 'today':
                cutoff = datetime.now().strftime('%Y-%m-%d')
                date_filter = " AND date(timestamp) = ?"
                date_params = [cutoff]
            elif date_range == 'week':
                cutoff = (datetime.now() - timedelta(days=7)).isoformat()
                date_filter = " AND timestamp >= ?"
                date_params = [cutoff]
            
            # Build base filter
            if session_id:
                base_filter = "WHERE session_id = ?" + date_filter
                params = [session_id] + date_params
            else:
                base_filter = ("WHERE 1=1" + date_filter) if date_filter else ""
                params = date_params
            
            # Timeline (last 60 minutes in 1-min buckets)
            cursor.execute(f"""
                SELECT 
                    strftime('%H:%M', timestamp) as minute,
                    SUM(CASE WHEN vehicle_type = 'commercial-vehicle' THEN 1 ELSE 0 END) as commercial,
                    SUM(CASE WHEN vehicle_type = 'military_vehicle' THEN 1 ELSE 0 END) as military,
                    SUM(CASE WHEN vehicle_type = 'gun' THEN 1 ELSE 0 END) as gun,
                    SUM(CASE WHEN vehicle_type = 'Grenade' THEN 1 ELSE 0 END) as grenade
                FROM detections
                {base_filter}
                GROUP BY minute
                ORDER BY minute DESC
                LIMIT 60
            """, params)
            timeline = [dict(row) for row in cursor.fetchall()]
            
            # Type distribution
            cursor.execute(f"""
                SELECT vehicle_type, COUNT(*) as count
                FROM detections
                {base_filter}
                GROUP BY vehicle_type
            """, params)
            type_dist = {row['vehicle_type']: row['count'] for row in cursor.fetchall()}
            
            # Confidence histogram (0.3-1.0 in 0.1 buckets)
            cursor.execute(f"""
                SELECT 
                    CAST(confidence * 10 AS INTEGER) / 10.0 as bucket,
                    COUNT(*) as count
                FROM detections
                {base_filter}
                GROUP BY bucket
                ORDER BY bucket
            """, params)
            conf_hist = {f"{row['bucket']:.1f}-{row['bucket']+0.1:.1f}": row['count'] 
                        for row in cursor.fetchall()}
            
            # Plate detection rate
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN plate_text IS NOT NULL AND plate_text != '' THEN 1 ELSE 0 END) as with_plate
                FROM detections
                {base_filter}
            """, params)
            row = cursor.fetchone()
            total = row['total'] or 1
            plate_rate = (row['with_plate'] or 0) / total
            
            # Hourly heatmap
            cursor.execute(f"""
                SELECT 
                    strftime('%w', timestamp) as day,
                    strftime('%H', timestamp) as hour,
                    COUNT(*) as count
                FROM detections
                {base_filter}
                GROUP BY day, hour
            """, params)
            days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
            heatmap = [{'day': days[int(row['day'])], 'hour': int(row['hour']), 'count': row['count']} 
                      for row in cursor.fetchall()]
            
            # Top plates - clean filter construction
            top_plates_filter = "WHERE plate_text IS NOT NULL AND plate_text != ''"
            top_plates_params = []
            if session_id:
                top_plates_filter += " AND session_id = ?"
                top_plates_params.append(session_id)
            if date_filter:
                top_plates_filter += date_filter
                top_plates_params.extend(date_params)
            
            cursor.execute(f"""
                SELECT 
                    plate_text,
                    vehicle_type,
                    COUNT(*) as count,
                    MAX(timestamp) as last_seen,
                    AVG(plate_conf) as avg_conf
                FROM detections
                {top_plates_filter}
                GROUP BY plate_text
                ORDER BY count DESC
                LIMIT 20
            """, top_plates_params)
            top_plates = [dict(row) for row in cursor.fetchall()]
            
            return {
                "timeline": timeline,
                "type_distribution": type_dist,
                "confidence_histogram": conf_hist,
                "plate_detection_rate": round(plate_rate, 3),
                "hourly_heatmap": heatmap,
                "top_plates": top_plates,
                "performance": {"avg_fps": 0, "min_fps": 0, "max_fps": 0}
            }
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of recent sessions with stats."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT 
                    session_id,
                    MIN(timestamp) as start_time,
                    MAX(timestamp) as end_time,
                    COUNT(*) as total_detections,
                    SUM(CASE WHEN vehicle_type = 'military_vehicle' THEN 1 ELSE 0 END) as military_count,
                    SUM(CASE WHEN vehicle_type = 'commercial-vehicle' THEN 1 ELSE 0 END) as commercial_count,
                    SUM(CASE WHEN vehicle_type = 'gun' THEN 1 ELSE 0 END) as gun_count,
                    SUM(CASE WHEN vehicle_type = 'Grenade' THEN 1 ELSE 0 END) as grenade_count,
                    SUM(CASE WHEN plate_text IS NOT NULL AND plate_text != '' THEN 1 ELSE 0 END) as plate_read_count
                FROM detections
                GROUP BY session_id
                ORDER BY start_time DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def clear_old_data(self, days: int = 7) -> int:
        """
        Delete detections older than specified days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of rows deleted
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM detections WHERE timestamp < ?", (cutoff,))
            count = cursor.fetchone()[0]
            
            if count > 0:
                cursor.execute("DELETE FROM detections WHERE timestamp < ?", (cutoff,))
                # Optimize database after delete
                cursor.execute("VACUUM")
            
            return count
    
    def start_session(self, session_id: str) -> None:
        """Record a new session start."""
        query = "INSERT OR REPLACE INTO sessions (id, start_time) VALUES (?, ?)"
        params = (session_id, datetime.now().isoformat())
        self._write_queue.put((query, params))
    
    def end_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        query = "UPDATE sessions SET end_time = ? WHERE id = ?"
        params = (datetime.now().isoformat(), session_id)
        self._write_queue.put((query, params))
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
        Run PRAGMA integrity_check on database.
        
        Returns:
            Dict with 'ok' bool and 'details' list.
            Never raises — system must keep running even if check fails.
        """
        try:
            with self._cursor() as cursor:
                cursor.execute("PRAGMA integrity_check")
                results = [row[0] for row in cursor.fetchall()]
                ok = results == ["ok"]
                return {"ok": ok, "details": results}
        except Exception as e:
            return {"ok": False, "details": [str(e)]}
    
    def close(self) -> None:
        """Shutdown database gracefully, flushing write queue."""
        self._shutdown.set()
        self._write_queue.put(None)  # Signal writer to stop
        
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)
        
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()


# Global database instance (lazy initialized)
_db: Optional[Database] = None


def get_database() -> Database:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def close_database() -> None:
    """Close the global database instance."""
    global _db
    if _db is not None:
        _db.close()
        _db = None
