"""
SENTINEL Detector Module

Two-stage vehicle detection pipeline with CLAHE preprocessing,
per-class NMS, and structured detection results.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
from ultralytics import YOLO

from config import get_config


@dataclass
class DetectionResult:
    """Structured detection result from the two-stage pipeline."""
    vehicle_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: Optional[int] = None
    plate_bbox: Optional[Tuple[int, int, int, int]] = None
    plate_conf: Optional[float] = None
    plate_text: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "vehicle_type": self.vehicle_type,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "track_id": self.track_id,
            "plate_bbox": self.plate_bbox,
            "plate_conf": round(self.plate_conf, 3) if self.plate_conf else None,
            "plate_text": self.plate_text,
            "timestamp": self.timestamp,
        }


class Detector:
    """Two-stage vehicle and plate detector with CLAHE preprocessing."""
    
    def __init__(self, vehicle_model_path: Optional[str] = None, 
                 plate_model_path: Optional[str] = None):
        """
        Initialize detector with YOLO models.
        
        Args:
            vehicle_model_path: Path to vehicle detection model
            plate_model_path: Path to license plate model
        """
        self.cfg = get_config()
        
        # Load models
        vehicle_path = vehicle_model_path or self.cfg.VEHICLE_MODEL_PATH
        plate_path = plate_model_path or self.cfg.PLATE_MODEL_PATH
        
        self.vehicle_model = YOLO(vehicle_path)
        self.plate_model = YOLO(plate_path)
        
        # CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        self._warmed_up = False
    
    def warmup(self) -> None:
        """
        Run warmup inference to pre-load CUDA/CPU kernels.
        Eliminates first-frame latency spike.
        """
        if self._warmed_up:
            return
        
        print("  🔥 Warming up detection models...")
        
        # Create blank frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run through both models
        self.vehicle_model.predict(
            source=dummy_frame,
            conf=0.5,
            verbose=False,
            imgsz=self.cfg.YOLO_IMGSZ
        )
        self.plate_model.predict(
            source=dummy_frame,
            conf=0.5,
            verbose=False,
            imgsz=self.cfg.YOLO_IMGSZ
        )
        
        self._warmed_up = True
        print("  ✅ Models warmed up")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame for better detection using CLAHE.
        
        Args:
            frame: BGR input frame
            
        Returns:
            Enhanced BGR frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l = self.clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """
        Run two-stage detection pipeline.
        
        Args:
            frame: BGR input frame
            
        Returns:
            Tuple of (annotated frame, list of DetectionResult)
        """
        detections: List[DetectionResult] = []
        annotated = frame.copy()
        h_orig, w_orig = frame.shape[:2]
        
        # Enhance frame
        enhanced = self.preprocess_frame(frame)
        
        # Downscale for inference
        scale = self.cfg.INFERENCE_WIDTH / w_orig
        if scale < 1.0:
            inf_frame = cv2.resize(enhanced, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_LINEAR)
        else:
            inf_frame = enhanced
            scale = 1.0
        
        # Get minimum confidence for initial detection
        min_vehicle_conf = min(self.cfg.class_conf.values()) if self.cfg.class_conf else 0.3
        
        # Stage 1: Vehicle detection
        vehicle_results = self.vehicle_model.predict(
            source=inf_frame,
            classes=list(self.cfg.TARGET_CLASSES),
            conf=min_vehicle_conf,
            verbose=False,
            imgsz=self.cfg.YOLO_IMGSZ,
            iou=self.cfg.NMS_IOU_DEFAULT,
            agnostic_nms=False,  # Per-class NMS
        )[0]
        
        for vbox in vehicle_results.boxes:
            vcls = int(vbox.cls[0])
            vconf = float(vbox.conf[0])
            vname = self.vehicle_model.names[vcls]
            
            # Per-class confidence filter
            required_conf = self.cfg.class_conf.get(vname, 0.5)
            if vconf < required_conf:
                continue
            
            # Scale coordinates back to original resolution
            vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])
            if scale < 1.0:
                vx1 = int(vx1 / scale)
                vy1 = int(vy1 / scale)
                vx2 = int(vx2 / scale)
                vy2 = int(vy2 / scale)
            
            # Clamp to frame bounds
            vx1 = max(0, vx1)
            vy1 = max(0, vy1)
            vx2 = min(w_orig, vx2)
            vy2 = min(h_orig, vy2)
            
            # Create detection result
            det = DetectionResult(
                vehicle_type=vname,
                confidence=vconf,
                bbox=(vx1, vy1, vx2, vy2),
            )
            
            # Draw vehicle box
            color = self.cfg.get_class_color(vname)
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), color, 2)
            
            # Draw label
            vlabel = f"{vname} {vconf:.2f}"
            (tw, th), _ = cv2.getTextSize(vlabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (vx1, vy1 - th - 8), (vx1 + tw + 4, vy1), color, -1)
            cv2.putText(annotated, vlabel, (vx1 + 2, vy1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.cfg.COLOR_TEXT_BG, 2)
            
            # Stage 2: License plate detection within vehicle bbox
            # Expand bbox slightly for better plate detection
            pad_x = int((vx2 - vx1) * 0.05)
            pad_y = int((vy2 - vy1) * 0.05)
            crop_x1 = max(0, vx1 - pad_x)
            crop_y1 = max(0, vy1 - pad_y)
            crop_x2 = min(w_orig, vx2 + pad_x)
            crop_y2 = min(h_orig, vy2 + pad_y)
            
            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                detections.append(det)
                continue
            
            # Enhance crop for plate detection
            crop_enhanced = self.preprocess_frame(crop)
            
            min_plate_conf = min(self.cfg.plate_conf.values()) if self.cfg.plate_conf else 0.25
            
            plate_results = self.plate_model.predict(
                source=crop_enhanced,
                conf=min_plate_conf,
                verbose=False,
                imgsz=self.cfg.YOLO_IMGSZ,
                iou=0.4,
                agnostic_nms=True,
            )[0]
            
            for pbox in plate_results.boxes:
                pconf = float(pbox.conf[0])
                pname = self.plate_model.names[int(pbox.cls[0])]
                
                # Per-class plate confidence filter
                required_plate_conf = self.cfg.plate_conf.get(pname, 0.3)
                if pconf < required_plate_conf:
                    continue
                
                px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                
                # Offset from crop to full frame
                abs_px1 = crop_x1 + px1
                abs_py1 = crop_y1 + py1
                abs_px2 = crop_x1 + px2
                abs_py2 = crop_y1 + py2
                
                # Update detection with plate info
                det.plate_bbox = (abs_px1, abs_py1, abs_px2, abs_py2)
                det.plate_conf = pconf
                
                # Draw plate box
                cv2.rectangle(annotated, (abs_px1, abs_py1), (abs_px2, abs_py2), 
                            self.cfg.COLOR_PLATE, 2)
                
                plabel = f"Plate {pconf:.2f}"
                (tw2, th2), _ = cv2.getTextSize(plabel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (abs_px1, abs_py2), 
                            (abs_px1 + tw2 + 4, abs_py2 + th2 + 8),
                            self.cfg.COLOR_PLATE, -1)
                cv2.putText(annotated, plabel, (abs_px1 + 2, abs_py2 + th2 + 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.cfg.COLOR_TEXT_BG, 2)
                
                break  # Only take first plate per vehicle
            
            detections.append(det)
        
        return annotated, detections
    
    def detect_and_annotate(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Alias for detect() for backward compatibility."""
        return self.detect(frame)


# Global detector instance (lazy initialized)
_detector: Optional[Detector] = None


def get_detector() -> Detector:
    """Get or create the global detector instance."""
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector
