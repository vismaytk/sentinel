"""
SENTINEL OCR Module

License plate text recognition using EasyOCR with lazy loading,
caching, and timeout mechanisms.
"""

import cv2
import logging
import numpy as np
import hashlib
import time
import threading
from typing import Optional, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from collections import deque

from config import get_config

logger = logging.getLogger(__name__)

# Try to import EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not installed - plate text recognition disabled")


class PlateOCR:
    """
    License plate OCR using EasyOCR with optimizations:
    - Lazy initialization to avoid startup delay
    - Caching to avoid redundant OCR on same plate
    - Timeout mechanism to skip slow OCR
    - Preprocessing for better accuracy
    """
    
    def __init__(self):
        """Initialize OCR (reader created lazily on first use)."""
        self.cfg = get_config()
        self._reader = None
        self._reader_lock = threading.Lock()
        self._cache: Dict[str, Tuple[Optional[str], Optional[float]]] = {}
        self._cache_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.cfg.OCR_WORKERS)
        self._disabled = False  # Permanent disable only when dependency missing
        self._next_init_retry_ts = 0.0
        self._logged_errors: set = set()  # Track logged exception types
        
        # Track OCR timing per track_id
        self._track_ocr_times: Dict[int, float] = {}
        self._track_ocr_bboxes: Dict[int, Tuple[int, int, int, int]] = {}
        self._last_global_ocr_time: float = 0.0
        self._last_global_bbox: Optional[Tuple[int, int, int, int]] = None
        self._ocr_inflight_lock = threading.Lock()
        self._duration_samples = deque(maxlen=20)
        self._warmed_up = False
        self._last_success_text: Optional[str] = None
        self._last_success_conf: Optional[float] = None
        self._last_success_bbox: Optional[Tuple[int, int, int, int]] = None
        self._last_success_ts: float = 0.0

    @staticmethod
    def _normalize_plate_text(raw: str) -> str:
        """Normalize OCR output to alphanumeric uppercase plate format."""
        if not raw:
            return ""
        cleaned = "".join(ch for ch in raw.upper() if ch.isalnum())
        # Typical plates are short alphanumeric strings; reject noise.
        if len(cleaned) < 4 or len(cleaned) > 12:
            return ""
        has_alpha = any(ch.isalpha() for ch in cleaned)
        has_digit = any(ch.isdigit() for ch in cleaned)
        if not (has_alpha and has_digit):
            return ""
        return cleaned
    
    @property
    def reader(self):
        """Lazily initialize EasyOCR reader."""
        if self._disabled:
            return None
        if time.time() < self._next_init_retry_ts:
            return None
        if self._reader is None:
            with self._reader_lock:
                if self._reader is None and EASYOCR_AVAILABLE and self.cfg.ENABLE_OCR:
                    try:
                        logger.info("Initializing EasyOCR reader...")
                        self._reader = easyocr.Reader(
                            list(self.cfg.OCR_LANGUAGES),
                            gpu=self.cfg.OCR_GPU,
                            verbose=False
                        )
                        logger.info("EasyOCR ready")
                    except Exception as e:
                        logger.warning("Failed to initialize EasyOCR: %s", e)
                        # Retry later in case failure was transient (model download, startup race).
                        self._next_init_retry_ts = time.time() + 30.0
                        return None
        return self._reader

    def warmup(self) -> bool:
        """Warm up OCR model to avoid first-inference latency spikes."""
        if self._warmed_up:
            return self.reader is not None
        r = self.reader
        if r is None:
            return False
        try:
            dummy = np.full((72, 220, 3), 255, dtype=np.uint8)
            cv2.putText(dummy, "MH12AB1234", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            _ = r.readtext(
                dummy,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            self._warmed_up = True
            return True
        except Exception as e:
            err_type = type(e).__name__
            if err_type not in self._logged_errors:
                logger.warning("OCR warmup error (%s): %s", err_type, e)
                self._logged_errors.add(err_type)
            return False
    
    def _compute_bbox_hash(self, plate_crop: np.ndarray) -> str:
        """Compute hash of plate crop for caching."""
        # Resize to small fixed size for consistent hashing
        small = cv2.resize(plate_crop, (64, 32))
        return hashlib.md5(small.tobytes()).hexdigest()
    
    def _should_run_ocr(self, track_id: Optional[int], bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if OCR should be run based on throttling rules.
        
        OCR is skipped if:
        - Bbox has moved less than OCR_MIN_MOVEMENT_PX from last OCR
        - Less than OCR_COOLDOWN_SEC has elapsed since last OCR on this track
        """
        if track_id is None:
            now = time.time()
            if now - self._last_global_ocr_time < self.cfg.OCR_COOLDOWN_SEC:
                last_bbox = self._last_global_bbox
                if last_bbox:
                    dx = abs((bbox[0] + bbox[2]) / 2 - (last_bbox[0] + last_bbox[2]) / 2)
                    dy = abs((bbox[1] + bbox[3]) / 2 - (last_bbox[1] + last_bbox[3]) / 2)
                    if max(dx, dy) < self.cfg.OCR_MIN_MOVEMENT_PX:
                        return False
            return True
        
        now = time.time()
        
        # Check cooldown
        last_time = self._track_ocr_times.get(track_id, 0)
        if now - last_time < self.cfg.OCR_COOLDOWN_SEC:
            # Check if bbox has moved significantly
            last_bbox = self._track_ocr_bboxes.get(track_id)
            if last_bbox:
                dx = abs((bbox[0] + bbox[2]) / 2 - (last_bbox[0] + last_bbox[2]) / 2)
                dy = abs((bbox[1] + bbox[3]) / 2 - (last_bbox[1] + last_bbox[3]) / 2)
                if max(dx, dy) < self.cfg.OCR_MIN_MOVEMENT_PX:
                    return False
        
        return True
    
    def _preprocess_plate(self, plate_crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess plate image for better OCR accuracy.
        
        Steps:
        1. Upscale 2x with INTER_CUBIC
        2. Convert to grayscale
        3. Apply adaptive thresholding
        4. Morphological closing to fill gaps
        5. Denoise
        
        Returns:
            Tuple of (binary image, inverted binary image) to try both
        """
        # Upscale 2x
        h, w = plate_crop.shape[:2]
        upscaled = cv2.resize(plate_crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(upscaled.shape) == 3:
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        else:
            gray = upscaled
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological closing to fill gaps in characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(closed, h=10)
        
        # Also prepare inverted version
        inverted = cv2.bitwise_not(denoised)
        
        return denoised, inverted
    
    def _run_ocr_single(self, preprocessed: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """Run OCR on a single preprocessed image."""
        if self.reader is None:
            return None, None
        
        results = self.reader.readtext(
            preprocessed,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            text_threshold=0.4,
            low_text=0.2,
            link_threshold=0.3,
        )
        
        if results:
            best_text = None
            best_conf = 0.0
            for (_, text, conf) in results:
                if conf < self.cfg.OCR_MIN_TEXT_CONF:
                    continue
                normalized = self._normalize_plate_text(text)
                if not normalized:
                    continue
                score = conf * min(len(normalized), 8)
                if best_text is None or score > best_conf:
                    best_text = normalized
                    best_conf = score
            if best_text:
                return best_text, float(best_conf / min(len(best_text), 8))
        
        return None, None
    
    def _run_ocr(self, preprocessed: np.ndarray, inverted: np.ndarray = None) -> Tuple[Optional[str], Optional[float]]:
        """Run OCR on preprocessed image, try both normal and inverted."""
        if self.reader is None:
            return None, None
        
        try:
            # Try normal version
            text1, conf1 = self._run_ocr_single(preprocessed)
            
            # Try inverted version if available
            if inverted is not None:
                text2, conf2 = self._run_ocr_single(inverted)
                
                # Return the one with higher confidence
                if text2 and (not text1 or (conf2 or 0) > (conf1 or 0)):
                    return text2, conf2
            
            return text1, conf1
            
        except Exception as e:
            # Log once per exception type
            err_type = type(e).__name__
            if err_type not in self._logged_errors:
                logger.warning("OCR error (%s): %s", err_type, e)
                self._logged_errors.add(err_type)
            return None, None

    def _run_ocr_variants(self, plate_crop: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Try OCR across multiple preprocessing variants and choose best result.
        """
        preprocessed, inverted = self._preprocess_plate(plate_crop)
        text, conf = self._run_ocr(preprocessed, inverted)
        if text and (conf or 0.0) >= 0.55:
            return text, conf

        # Raw grayscale variant helps when adaptive threshold over-processes text
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
        gray_up = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        text2, conf2 = self._run_ocr_single(gray_up)
        if text2 and (not text or (conf2 or 0.0) > (conf or 0.0)):
            return text2, conf2
        return text, conf
    
    def read_plate(
        self,
        frame: np.ndarray,
        plate_bbox: Tuple[int, int, int, int],
        track_id: Optional[int] = None
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Read text from license plate region.
        
        Args:
            frame: Full frame image
            plate_bbox: (x1, y1, x2, y2) of plate region
            track_id: Optional track ID for throttling
            
        Returns:
            Tuple of (text, confidence) or (None, None) if not readable
        """
        if self._disabled:
            return None, None
            
        if not self.cfg.ENABLE_OCR or not EASYOCR_AVAILABLE:
            return None, None
        
        # Extract plate crop
        x1, y1, x2, y2 = plate_bbox
        
        # Validate bbox
        if x2 <= x1 or y2 <= y1:
            return None, None
        
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        plate_crop = frame[y1:y2, x1:x2]
        
        if plate_crop.size == 0:
            return None, None
        
        # Expand crop slightly for more OCR context
        pad_x = max(2, int((x2 - x1) * 0.10))
        pad_y = max(2, int((y2 - y1) * 0.15))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size == 0:
            return None, None

        # Check cache by image hash
        crop_hash = self._compute_bbox_hash(plate_crop)
        with self._cache_lock:
            cached = self._cache.get(crop_hash)
            if cached and cached[0]:
                return cached

        # Check throttling after cache lookup so recent OCR can still propagate.
        if not self._should_run_ocr(track_id, plate_bbox):
            if track_id is not None:
                cache_key = f"track_{track_id}"
                with self._cache_lock:
                    cached_track = self._cache.get(cache_key)
                    if cached_track and cached_track[0]:
                        return cached_track
            if (
                self._last_success_text
                and self._last_success_bbox
                and (time.time() - self._last_success_ts) <= max(3.0, self.cfg.OCR_COOLDOWN_SEC * 2)
            ):
                bx = (plate_bbox[0] + plate_bbox[2]) / 2
                by = (plate_bbox[1] + plate_bbox[3]) / 2
                lx = (self._last_success_bbox[0] + self._last_success_bbox[2]) / 2
                ly = (self._last_success_bbox[1] + self._last_success_bbox[3]) / 2
                if max(abs(bx - lx), abs(by - ly)) <= max(24, self.cfg.OCR_MIN_MOVEMENT_PX * 3):
                    return self._last_success_text, self._last_success_conf
            return None, None

        # Prevent OCR queue pile-up when tracking is disabled or frame rate is high.
        if not self._ocr_inflight_lock.acquire(blocking=False):
            if track_id is not None:
                cache_key = f"track_{track_id}"
                with self._cache_lock:
                    cached_track = self._cache.get(cache_key)
                    if cached_track and cached_track[0]:
                        return cached_track
            return None, None

        # Run OCR with adaptive timeout (based on recent OCR durations).
        timeout_sec = max(0.5, self.cfg.OCR_TIMEOUT_MS / 1000.0)
        if self._duration_samples:
            p90_estimate = sorted(self._duration_samples)[int(len(self._duration_samples) * 0.9)]
            timeout_sec = max(timeout_sec, min(8.0, p90_estimate * 1.5))
        else:
            timeout_sec = max(timeout_sec, 6.0)

        # Run OCR with timeout
        try:
            started = time.time()
            future = self._executor.submit(self._run_ocr_variants, plate_crop)
            text, conf = future.result(timeout=timeout_sec)
            self._duration_samples.append(time.time() - started)
        except FuturesTimeoutError:
            future.cancel()
            return None, None
        except Exception as e:
            err_type = type(e).__name__
            if err_type not in self._logged_errors:
                logger.warning("OCR error in read_plate (%s): %s", err_type, e)
                self._logged_errors.add(err_type)
            return None, None
        finally:
            self._last_global_ocr_time = time.time()
            self._last_global_bbox = plate_bbox
            self._ocr_inflight_lock.release()
        
        # Update cache
        with self._cache_lock:
            # Cache positive OCR only; avoid poisoning cache with transient failures.
            if text:
                self._cache[crop_hash] = (text, conf)
                if track_id is not None:
                    self._cache[f"track_{track_id}"] = (text, conf)
                self._last_success_text = text
                self._last_success_conf = conf
                self._last_success_bbox = plate_bbox
                self._last_success_ts = time.time()
        
        # Update tracking info
        if track_id is not None:
            self._track_ocr_times[track_id] = time.time()
            self._track_ocr_bboxes[track_id] = plate_bbox
        
        return text, conf
    
    def clear_cache(self):
        """Clear OCR cache."""
        with self._cache_lock:
            self._cache.clear()
        self._track_ocr_times.clear()
        self._track_ocr_bboxes.clear()
    
    def close(self):
        """Shutdown OCR resources."""
        self._executor.shutdown(wait=False)


# Global OCR instance
_ocr: Optional[PlateOCR] = None


def get_ocr() -> PlateOCR:
    """Get or create the global OCR instance."""
    global _ocr
    if _ocr is None:
        _ocr = PlateOCR()
    return _ocr
