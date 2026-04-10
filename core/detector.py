"""
SENTINEL Detector Module

Two-stage vehicle detection pipeline with CLAHE preprocessing,
per-class NMS, Platt scaling calibration, optional TTA, and 
multi-scale inference support.
"""

import cv2
import logging
import numpy as np
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from ultralytics import YOLO

from config import get_config

logger = logging.getLogger(__name__)


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
    raw_confidence: float = 0.0  # Pre-calibration confidence
    detection_category: str = "vehicle"  # "vehicle" or "weapon"
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.raw_confidence == 0.0:
            self.raw_confidence = self.confidence
        # Set category based on detection type (weapons vs vehicles)
        if self.vehicle_type in ("gun", "Grenade"):
            self.detection_category = "weapon"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "vehicle_type": self.vehicle_type,
            "confidence": round(self.confidence, 3),
            "raw_confidence": round(self.raw_confidence, 3),
            "bbox": self.bbox,
            "track_id": self.track_id,
            "plate_bbox": self.plate_bbox,
            "plate_conf": round(self.plate_conf, 3) if self.plate_conf else None,
            "plate_text": self.plate_text,
            "timestamp": self.timestamp,
            "detection_category": self.detection_category,
        }


class Detector:
    """Two-stage vehicle and plate detector with CLAHE preprocessing,
    TTA, multi-scale inference, and Platt calibration."""
    
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
        
        logger.info("Warming up models...")
        
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
        logger.info("Models warmed up")
    
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
    
    def _apply_gamma(self, frame: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """Apply gamma correction for brightness adjustment."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in range(256)]).astype("uint8")
        return cv2.LUT(frame, table)
    
    def _platt_calibrate(self, conf: float, class_name: str) -> float:
        """Apply Platt scaling calibration to confidence score."""
        a = self.cfg.PLATT_A.get(class_name, 2.0)
        b = self.cfg.PLATT_B.get(class_name, -1.0)
        return 1.0 / (1.0 + math.exp(-(a * conf + b)))
    
    def _compute_iou(self, box1: tuple, box2: tuple) -> float:
        """Compute IoU between two boxes (x1, y1, x2, y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / max(union, 1e-6)
    
    def _wbf_merge(self, boxes_list: List[List], scores_list: List[List], 
                   labels_list: List[List], iou_thr: float = 0.55) -> Tuple[List, List, List]:
        """
        Weighted Boxes Fusion - pure numpy implementation.
        Clusters boxes by IoU, averages coords weighted by score.
        
        Args:
            boxes_list: List of box arrays from different models/augmentations
            scores_list: List of score arrays
            labels_list: List of label arrays
            iou_thr: IoU threshold for clustering
            
        Returns:
            Tuple of (fused_boxes, fused_scores, fused_labels)
        """
        if not boxes_list or all(len(b) == 0 for b in boxes_list):
            return [], [], []
        
        # Flatten all inputs
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
            for box, score, label in zip(boxes, scores, labels):
                all_boxes.append(box)
                all_scores.append(score)
                all_labels.append(label)
        
        if not all_boxes:
            return [], [], []
        
        # Sort by score descending
        indices = np.argsort(all_scores)[::-1]
        all_boxes = [all_boxes[i] for i in indices]
        all_scores = [all_scores[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
        
        # Cluster by IoU
        clusters = []
        used = [False] * len(all_boxes)
        
        for i in range(len(all_boxes)):
            if used[i]:
                continue
            
            cluster = [(all_boxes[i], all_scores[i], all_labels[i])]
            used[i] = True
            
            for j in range(i + 1, len(all_boxes)):
                if used[j] or all_labels[j] != all_labels[i]:
                    continue
                
                if self._compute_iou(all_boxes[i], all_boxes[j]) >= iou_thr:
                    cluster.append((all_boxes[j], all_scores[j], all_labels[j]))
                    used[j] = True
            
            clusters.append(cluster)
        
        # Fuse each cluster
        fused_boxes = []
        fused_scores = []
        fused_labels = []
        
        for cluster in clusters:
            if not cluster:
                continue
            
            # Weighted average of box coords
            total_weight = sum(c[1] for c in cluster)
            x1 = sum(c[0][0] * c[1] for c in cluster) / total_weight
            y1 = sum(c[0][1] * c[1] for c in cluster) / total_weight
            x2 = sum(c[0][2] * c[1] for c in cluster) / total_weight
            y2 = sum(c[0][3] * c[1] for c in cluster) / total_weight
            
            # Average score boosted by cluster size
            avg_score = total_weight / len(cluster) * min(len(cluster), 3) / 3
            avg_score = min(avg_score, 1.0)
            
            fused_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            fused_scores.append(avg_score)
            fused_labels.append(cluster[0][2])  # Use first label
        
        return fused_boxes, fused_scores, fused_labels
    
    def _run_vehicle_detection(self, frame: np.ndarray, scale: float, 
                               imgsz: int, min_conf: float) -> List[Tuple]:
        """Run vehicle detection at a specific scale. Returns list of (box, conf, name)."""
        results = []
        
        h_orig, w_orig = frame.shape[:2]
        
        vehicle_results = self.vehicle_model.predict(
            source=frame,
            classes=list(self.cfg.get_active_classes()),
            conf=min_conf,
            verbose=False,
            imgsz=imgsz,
            iou=self.cfg.NMS_IOU_DEFAULT,
            agnostic_nms=False,
        )[0]
        
        for vbox in vehicle_results.boxes:
            vcls = int(vbox.cls[0])
            vconf = float(vbox.conf[0])
            vname = self.vehicle_model.names[vcls]
            
            vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])
            if scale < 1.0:
                vx1 = int(vx1 / scale)
                vy1 = int(vy1 / scale)
                vx2 = int(vx2 / scale)
                vy2 = int(vy2 / scale)
            
            results.append(((vx1, vy1, vx2, vy2), vconf, vname))
        
        return results
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """
        Run two-stage detection pipeline with optional TTA and multi-scale.
        
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
        
        # Collect detections from all methods
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # Primary detection at main scale
        primary_results = self._run_vehicle_detection(
            inf_frame, scale, self.cfg.YOLO_IMGSZ, min_vehicle_conf
        )
        boxes_main = [r[0] for r in primary_results]
        scores_main = [r[1] for r in primary_results]
        labels_main = [r[2] for r in primary_results]
        all_boxes.append(boxes_main)
        all_scores.append(scores_main)
        all_labels.append(labels_main)
        
        # Multi-scale inference
        if self.cfg.MULTI_SCALE_INFERENCE:
            secondary_imgsz = max(320, self.cfg.YOLO_IMGSZ - self.cfg.MULTI_SCALE_OFFSET)
            secondary_results = self._run_vehicle_detection(
                inf_frame, scale, secondary_imgsz, min_vehicle_conf
            )
            all_boxes.append([r[0] for r in secondary_results])
            all_scores.append([r[1] for r in secondary_results])
            all_labels.append([r[2] for r in secondary_results])
        
        # Test-Time Augmentation
        if self.cfg.ENABLE_TTA:
            # Horizontal flip
            flipped = cv2.flip(inf_frame, 1)
            flip_results = self._run_vehicle_detection(
                flipped, scale, self.cfg.YOLO_IMGSZ, min_vehicle_conf
            )
            # Mirror boxes back
            flip_w = inf_frame.shape[1]
            boxes_flip = []
            for (x1, y1, x2, y2), conf, name in flip_results:
                # Account for scaling
                if scale < 1.0:
                    flip_w_orig = w_orig
                else:
                    flip_w_orig = flip_w
                boxes_flip.append((flip_w_orig - x2, y1, flip_w_orig - x1, y2))
            all_boxes.append(boxes_flip)
            all_scores.append([r[1] for r in flip_results])
            all_labels.append([r[2] for r in flip_results])
            
            # Gamma-brightened
            bright = self._apply_gamma(inf_frame, gamma=1.2)
            bright_results = self._run_vehicle_detection(
                bright, scale, self.cfg.YOLO_IMGSZ, min_vehicle_conf
            )
            all_boxes.append([r[0] for r in bright_results])
            all_scores.append([r[1] for r in bright_results])
            all_labels.append([r[2] for r in bright_results])
        
        # Merge using WBF if TTA or multi-scale is enabled
        if self.cfg.ENABLE_TTA or self.cfg.MULTI_SCALE_INFERENCE:
            fused_boxes, fused_scores, fused_labels = self._wbf_merge(
                all_boxes, all_scores, all_labels, iou_thr=0.55
            )
        else:
            fused_boxes = boxes_main
            fused_scores = scores_main
            fused_labels = labels_main
        
        # Process fused detections
        for bbox, vconf, vname in zip(fused_boxes, fused_scores, fused_labels):
            vx1, vy1, vx2, vy2 = bbox
            
            # Per-class confidence filter
            required_conf = self.cfg.class_conf.get(vname, 0.5)
            if vconf < required_conf:
                continue
            
            # Clamp to frame bounds
            vx1 = max(0, vx1)
            vy1 = max(0, vy1)
            vx2 = min(w_orig, vx2)
            vy2 = min(h_orig, vy2)
            
            # Apply Platt calibration
            raw_conf = vconf
            calibrated_conf = self._platt_calibrate(vconf, vname)
            
            # Create detection result
            det = DetectionResult(
                vehicle_type=vname,
                confidence=calibrated_conf,
                raw_confidence=raw_conf,
                bbox=(vx1, vy1, vx2, vy2),
            )
            
            # Draw vehicle box
            color = self.cfg.get_class_color(vname)
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), color, 2)
            
            # Draw label with calibrated confidence
            vlabel = f"{vname} {calibrated_conf:.2f}"
            (tw, th), _ = cv2.getTextSize(vlabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (vx1, vy1 - th - 8), (vx1 + tw + 4, vy1), color, -1)
            cv2.putText(annotated, vlabel, (vx1 + 2, vy1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.cfg.COLOR_TEXT_BG, 2)
            
            # Skip plate detection for weapons — no plates on guns/grenades
            if det.detection_category == "weapon":
                detections.append(det)
                continue
            
            # Stage 2: License plate detection within vehicle bbox
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
