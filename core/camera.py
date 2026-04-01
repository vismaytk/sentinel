"""
SENTINEL Camera Module

Enhanced camera stream with frame queue, metadata, and reconnection logic.
Supports IP cameras, local files, and webcams.
"""

import cv2
import os
import time
import threading
import numpy as np
from collections import deque
from datetime import datetime
from typing import Optional, Tuple

from config import get_config


class CameraStream:
    """
    Threaded camera stream with frame queue and metadata.
    
    Features:
    - Supports IP camera URL, local file path, or webcam index
    - Frame queue to prevent lag buildup
    - Automatic reconnection on failure
    - Frame timestamp and index tracking
    """
    
    def __init__(self, source: Optional[str] = None):
        """
        Initialize camera stream.
        
        Args:
            source: IP camera URL, file path, or webcam index (as string).
                   Defaults to config IP_CAM_URL.
        """
        self.cfg = get_config()
        self._source = source or self.cfg.IP_CAM_URL
        
        # Parse source type
        self._source_type = self._detect_source_type(self._source)
        
        # State
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_buffer: deque = deque(maxlen=self.cfg.FRAME_BUFFER_SIZE)
        self._frame_lock = threading.Lock()
        self._connected = False
        self._running = True
        self._retry_count = 0
        
        # Metadata
        self._frames_read = 0
        self._frame_timestamp: Optional[datetime] = None
        self._resolution: Tuple[int, int] = (0, 0)
        self._source_fps: float = 0.0
        
        # Status tracking
        self._status = "connecting"
        self._status_lock = threading.Lock()
        
        # Start reader thread
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
    
    def _detect_source_type(self, source: str) -> str:
        """Detect whether source is URL, file, or webcam."""
        if source.startswith(("http://", "https://", "rtsp://")):
            return "ip_camera"
        elif source.isdigit():
            return "webcam"
        elif os.path.isfile(source):
            return "file"
        else:
            # Assume IP camera URL without protocol
            return "ip_camera"
    
    def _try_connect(self) -> bool:
        """Attempt to connect to camera source."""
        try:
            if self.cap is not None:
                self.cap.release()
            
            # Configure FFmpeg timeout for IP cameras
            if self._source_type == "ip_camera":
                timeout_us = self.cfg.CAMERA_TIMEOUT_MS
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    f"timeout;{timeout_us}|stimeout;{timeout_us}"
                )
                cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)
            elif self._source_type == "webcam":
                cap = cv2.VideoCapture(int(self._source))
            else:
                cap = cv2.VideoCapture(self._source)
            
            # Set buffer size to minimize latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    self._connected = True
                    self._retry_count = 0
                    
                    # Get metadata
                    self._resolution = (
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    )
                    self._source_fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Store first frame
                    with self._frame_lock:
                        self._frame_buffer.append(frame)
                        self._frame_timestamp = datetime.now()
                        self._frames_read = 1
                    
                    self._set_status("connected")
                    print(f"  ✅ Camera connected: {self._source}")
                    print(f"     Resolution: {self._resolution[0]}x{self._resolution[1]}")
                    return True
                else:
                    cap.release()
            else:
                cap.release()
                
        except Exception as e:
            print(f"  ⚠ Connection attempt failed: {e}")
        
        self._connected = False
        self._retry_count += 1
        self._set_status("connecting")
        return False
    
    def _read_loop(self):
        """Main camera reading loop (runs in thread)."""
        # Initial connection with retries
        while self._running and not self._connected:
            print(f"  📡 Connecting to camera (attempt {self._retry_count + 1})...")
            if self._try_connect():
                break
            # Exponential backoff: 0s, 3s, 6s, 9s, max 10s
            wait = min(3 * self._retry_count, 10)
            time.sleep(wait)
        
        # Main read loop
        while self._running:
            if not self._connected or self.cap is None:
                print(f"  📡 Reconnecting (attempt {self._retry_count + 1})...")
                self._set_status("connecting")
                if not self._try_connect():
                    time.sleep(3)
                    continue
            
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                if self._source_type == "file":
                    # Loop video file
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("  ⚠ Frame read failed — reconnecting...")
                    self._connected = False
                    self._set_status("error")
                    time.sleep(1)
                    continue
            
            # Update frame buffer
            with self._frame_lock:
                self._frame_buffer.append(frame)
                self._frame_timestamp = datetime.now()
                self._frames_read += 1
    
    def _set_status(self, status: str):
        """Thread-safe status update."""
        with self._status_lock:
            self._status = status
    
    @property
    def status(self) -> str:
        """Current camera status: connecting, connected, error."""
        with self._status_lock:
            return self._status
    
    @property
    def frame_timestamp(self) -> Optional[datetime]:
        """Timestamp of the latest frame."""
        with self._frame_lock:
            return self._frame_timestamp
    
    @property
    def frames_read(self) -> int:
        """Total number of frames read."""
        with self._frame_lock:
            return self._frames_read
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Camera resolution (width, height)."""
        return self._resolution
    
    @property
    def source(self) -> str:
        """Camera source URL/path."""
        return self._source
    
    @property
    def source_fps(self) -> float:
        """Native FPS of camera source."""
        return self._source_fps
    
    def is_connected(self) -> bool:
        """Check if camera is connected."""
        return self._connected
    
    def read(self) -> Optional[np.ndarray]:
        """
        Get the latest frame.
        
        Returns:
            Copy of latest frame, or None if no frame available.
        """
        with self._frame_lock:
            if len(self._frame_buffer) > 0:
                return self._frame_buffer[-1].copy()
            return None
    
    def get_frame_with_meta(self) -> Tuple[Optional[np.ndarray], Optional[datetime], int]:
        """
        Get frame with metadata.
        
        Returns:
            Tuple of (frame, timestamp, frame_index)
        """
        with self._frame_lock:
            if len(self._frame_buffer) > 0:
                return (
                    self._frame_buffer[-1].copy(),
                    self._frame_timestamp,
                    self._frames_read
                )
            return None, None, 0
    
    def get_info(self) -> dict:
        """Get camera info for status display."""
        return {
            "source": self._source,
            "source_type": self._source_type,
            "status": self.status,
            "connected": self._connected,
            "resolution": f"{self._resolution[0]}x{self._resolution[1]}",
            "source_fps": round(self._source_fps, 1),
            "frames_read": self._frames_read,
        }
    
    def stop(self):
        """Stop camera stream and release resources."""
        self._running = False
        
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def make_placeholder(text: str = "Connecting to camera...", w: int = 640, h: int = 360) -> np.ndarray:
    """
    Generate a placeholder frame when camera is not connected.
    
    Args:
        text: Message to display
        w: Frame width
        h: Frame height
        
    Returns:
        Placeholder frame as numpy array
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (13, 8, 4)  # Dark background matching SENTINEL theme
    
    # Add text
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (184, 163, 148), 2)
    
    # Add border
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (53, 37, 26), 1)
    
    return img


# Global camera instance
_camera: Optional[CameraStream] = None


def get_camera(source: Optional[str] = None) -> CameraStream:
    """Get or create the global camera instance."""
    global _camera
    if _camera is None:
        _camera = CameraStream(source)
    return _camera


def stop_camera():
    """Stop the global camera instance."""
    global _camera
    if _camera is not None:
        _camera.stop()
        _camera = None
