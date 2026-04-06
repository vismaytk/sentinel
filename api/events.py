"""
SENTINEL Events Module

Server-Sent Events (SSE) for real-time push to frontend.
Eliminates polling latency and reduces server load.
"""

import json
import logging
import time
import threading
from typing import Generator, Callable, Any, Optional

from flask import Response

from config import get_config

logger = logging.getLogger(__name__)


class SSEManager:
    """
    Server-Sent Events manager for real-time data push.
    
    Maintains a snapshot of latest stats and pushes to all connected clients.
    Includes heartbeat for connection keep-alive through proxies.
    """
    
    def __init__(self):
        """Initialize SSE manager."""
        self.cfg = get_config()
        self._stats: dict = {}
        self._lock = threading.Lock()
        self._data_provider: Optional[Callable[[], dict]] = None
    
    def set_data_provider(self, provider: Callable[[], dict]) -> None:
        """
        Set the data provider function.
        
        Args:
            provider: Callable that returns current stats dict
        """
        self._data_provider = provider
    
    def update_stats(self, stats: dict) -> None:
        """
        Update the stats snapshot.
        
        Args:
            stats: New stats dictionary
        """
        with self._lock:
            self._stats = stats.copy()
    
    def get_stats(self) -> dict:
        """Get current stats snapshot."""
        if self._data_provider:
            return self._data_provider()
        with self._lock:
            return self._stats.copy()
    
    def generate_events(self) -> Generator[str, None, None]:
        """
        Generator for SSE stream.
        
        Yields:
            SSE formatted strings with data and heartbeat pings
        """
        last_heartbeat = time.time()
        
        while True:
            # Get current data
            data = self.get_stats()
            
            # Yield data event
            yield f"data: {json.dumps(data)}\n\n"
            
            # Send heartbeat comment if needed (keeps connection alive through proxies)
            now = time.time()
            if now - last_heartbeat >= self.cfg.SSE_HEARTBEAT_INTERVAL:
                yield ": ping\n\n"
                last_heartbeat = now
            
            # Wait before next push
            time.sleep(self.cfg.SSE_PUSH_INTERVAL)
    
    def create_response(self) -> Response:
        """
        Create Flask SSE response.
        
        Returns:
            Flask Response with SSE stream
        """
        return Response(
            self.generate_events(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # Disable nginx buffering
                'Connection': 'keep-alive',
            }
        )


# Global SSE manager
_sse: Optional[SSEManager] = None


def get_sse_manager() -> SSEManager:
    """Get or create the global SSE manager."""
    global _sse
    if _sse is None:
        _sse = SSEManager()
    return _sse
