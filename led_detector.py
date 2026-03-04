#!/usr/bin/env python3
"""
LED Detector - Auto-detect LED regions using AI models.
Supports: Zero-shot (OWL-ViT) and Fine-tuned YOLO.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import json


@dataclass
class Detection:
    """Detected LED region."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    label: str = "led"
    
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)
    
    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


class LEDDetector:
    """Base class for LED detection."""
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError
    
    def detect_in_video(self, video_path: str, sample_frames: int = 10) -> List[Detection]:
        """Detect LEDs across multiple frames and merge results."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        all_detections = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            detections = self.detect(frame)
            all_detections.extend(detections)
        
        cap.release()
        
        # Merge overlapping detections
        if all_detections:
            return self._merge_detections(all_detections)
        return []
    
    def _merge_detections(self, detections: List[Detection], 
                          iou_threshold: float = 0.3) -> List[Detection]:
        """Merge overlapping detections using NMS-like approach."""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        merged = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
            
            # Find overlapping detections
            overlapping = [det]
            for j, other in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                if self._iou(det, other) > iou_threshold:
                    overlapping.append(other)
                    used.add(j)
            
            # Average the bounding boxes
            avg_x = int(np.mean([d.x for d in overlapping]))
            avg_y = int(np.mean([d.y for d in overlapping]))
            avg_w = int(np.mean([d.w for d in overlapping]))
            avg_h = int(np.mean([d.h for d in overlapping]))
            max_conf = max(d.confidence for d in overlapping)
            
            merged.append(Detection(avg_x, avg_y, avg_w, avg_h, max_conf))
            used.add(i)
        
        return merged
    
    def _iou(self, a: Detection, b: Detection) -> float:
        """Calculate Intersection over Union."""
        ax1, ay1, ax2, ay2 = a.as_xyxy()
        bx1, by1, bx2, by2 = b.as_xyxy()
        
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - intersection
        
        return intersection / union if union > 0 else 0.0


class OWLViTDetector(LEDDetector):
    """Zero-shot LED detection using OWL-ViT."""
    
    def __init__(self, prompts: List[str] = None, threshold: float = 0.1):
        self.prompts = prompts or [
            "LED light",
            "indicator light", 
            "glowing LED strip",
            "illuminated button",
            "light indicator on device"
        ]
        self.threshold = threshold
        self.processor = None
        self.model = None
        self._loaded = False
    
    def _load_model(self):
        if self._loaded:
            return
        
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            import torch
            
            print("Loading OWL-ViT model...")
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self._loaded = True
            print(f"OWL-ViT loaded on {self.device}")
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        self._load_model()
        import torch
        from PIL import Image
        
        # Convert BGR to RGB
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Process
        inputs = self.processor(text=[self.prompts], images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]
        
        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            box = box.cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            detections.append(Detection(
                x=x1, y=y1, w=x2-x1, h=y2-y1,
                confidence=float(score),
                label=self.prompts[label]
            ))
        
        return detections


class YOLODetector(LEDDetector):
    """Fine-tuned YOLO for LED detection."""
    
    def __init__(self, model_path: str = None, threshold: float = 0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self._loaded = False
    
    def _load_model(self):
        if self._loaded:
            return
        
        try:
            from ultralytics import YOLO
            
            if self.model_path and Path(self.model_path).exists():
                print(f"Loading custom YOLO model: {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                # Use pre-trained YOLOv8 (won't detect LEDs specifically)
                print("Loading YOLOv8n (pre-trained, no LED class)")
                self.model = YOLO("yolov8n.pt")
            
            self._loaded = True
        except ImportError:
            raise ImportError("Install ultralytics: pip install ultralytics")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        self._load_model()
        
        results = self.model(frame, conf=self.threshold, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = results.names[cls]
            
            detections.append(Detection(
                x=x1, y=y1, w=x2-x1, h=y2-y1,
                confidence=conf,
                label=label
            ))
        
        return detections


class HybridDetector(LEDDetector):
    """
    Hybrid approach: Use brightness detection + optional AI refinement.
    Falls back to traditional CV if AI models aren't available.
    """
    
    def __init__(self, use_ai: bool = True):
        self.use_ai = use_ai
        self.ai_detector = None
        
        if use_ai:
            try:
                self.ai_detector = OWLViTDetector()
            except ImportError:
                print("AI models not available, using CV-only detection")
                self.use_ai = False
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        # Always run CV detection (fast)
        cv_detections = self._detect_cv(frame)
        
        if not self.use_ai or self.ai_detector is None:
            return cv_detections
        
        # Optionally refine with AI
        ai_detections = self.ai_detector.detect(frame)
        
        # Merge: prefer AI detections, fall back to CV
        if ai_detections:
            return ai_detections
        return cv_detections
    
    def _detect_cv(self, frame: np.ndarray) -> List[Detection]:
        """Traditional CV-based LED detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Find bright, saturated regions
        mask = (v > 200) & (s > 80)
        mask = mask.astype(np.uint8) * 255
        
        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Too small
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            detections.append(Detection(
                x=x, y=y, w=w, h=h,
                confidence=min(1.0, area / 5000),  # Rough confidence
                label="led_cv"
            ))
        
        return detections


# ============================================================================
# YOLO Training Pipeline
# ============================================================================

class YOLOTrainer:
    """Train custom YOLO model for LED detection."""
    
    def __init__(self, data_dir: str = "led_dataset"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        
    def setup_dataset(self):
        """Create dataset directory structure."""
        for split in ["train", "val"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        print(f"Dataset structure created at {self.data_dir}")
        print("""
Next steps:
1. Add images to {data_dir}/images/train/ and {data_dir}/images/val/
2. Add YOLO labels to {data_dir}/labels/train/ and {data_dir}/labels/val/
   Label format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
3. Run trainer.train()

Or use trainer.annotate_from_video() to create training data from existing videos.
""")
    
    def annotate_from_video(self, video_path: str, output_prefix: str,
                            detector: LEDDetector = None, 
                            sample_frames: int = 20,
                            split: str = "train"):
        """
        Extract frames and auto-annotate using existing detector.
        Human should verify/correct annotations.
        """
        if detector is None:
            detector = HybridDetector(use_ai=False)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        
        images_out = self.images_dir / split
        labels_out = self.labels_dir / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect LEDs
            detections = detector.detect(frame)
            
            # Save image
            img_name = f"{output_prefix}_{i:04d}.jpg"
            cv2.imwrite(str(images_out / img_name), frame)
            
            # Save YOLO format labels
            label_name = f"{output_prefix}_{i:04d}.txt"
            with open(labels_out / label_name, 'w') as f:
                for det in detections:
                    # Convert to YOLO format (normalized)
                    x_center = (det.x + det.w / 2) / width
                    y_center = (det.y + det.h / 2) / height
                    w_norm = det.w / width
                    h_norm = det.h / height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        cap.release()
        print(f"Extracted {sample_frames} frames to {images_out}")
        print(f"Auto-generated labels in {labels_out}")
        print("⚠️  Review and correct labels before training!")
    
    def create_yaml(self):
        """Create YOLO dataset config file."""
        yaml_content = f"""
path: {self.data_dir.absolute()}
train: images/train
val: images/val

names:
  0: led
"""
        yaml_path = self.data_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created {yaml_path}")
        return yaml_path
    
    def train(self, epochs: int = 50, imgsz: int = 640, batch: int = 16):
        """Train YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Install ultralytics: pip install ultralytics")
        
        yaml_path = self.create_yaml()
        
        # Start with YOLOv8 nano (small, fast)
        model = YOLO("yolov8n.pt")
        
        print(f"Training YOLO on {self.data_dir}...")
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name="led_detector"
        )
        
        # Best model saved to runs/detect/led_detector/weights/best.pt
        print(f"\nTraining complete!")
        print(f"Best model: runs/detect/led_detector/weights/best.pt")
        return results


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LED Detection with AI")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect LEDs in video/image")
    detect_parser.add_argument("input", help="Video or image path")
    detect_parser.add_argument("--method", choices=["cv", "owlvit", "yolo", "hybrid"], 
                               default="hybrid", help="Detection method")
    detect_parser.add_argument("--model", help="Path to custom YOLO model")
    detect_parser.add_argument("--output", "-o", help="Output JSON path")
    detect_parser.add_argument("--visualize", "-v", action="store_true",
                               help="Save visualization image")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train custom YOLO model")
    train_parser.add_argument("--data-dir", default="led_dataset", help="Dataset directory")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--setup", action="store_true", help="Setup dataset structure")
    
    # Annotate command
    annotate_parser = subparsers.add_parser("annotate", help="Auto-annotate video for training")
    annotate_parser.add_argument("video", help="Video path")
    annotate_parser.add_argument("--prefix", default="frame", help="Output prefix")
    annotate_parser.add_argument("--frames", type=int, default=20, help="Frames to extract")
    annotate_parser.add_argument("--data-dir", default="led_dataset", help="Dataset directory")
    
    args = parser.parse_args()
    
    if args.command == "detect":
        # Select detector
        if args.method == "cv":
            detector = HybridDetector(use_ai=False)
        elif args.method == "owlvit":
            detector = OWLViTDetector()
        elif args.method == "yolo":
            detector = YOLODetector(model_path=args.model)
        else:
            detector = HybridDetector(use_ai=True)
        
        # Detect
        if args.input.lower().endswith(('.mp4', '.mov', '.avi')):
            detections = detector.detect_in_video(args.input)
        else:
            frame = cv2.imread(args.input)
            detections = detector.detect(frame)
        
        # Output
        result = {
            "input": args.input,
            "method": args.method,
            "detections": [
                {"x": d.x, "y": d.y, "w": d.w, "h": d.h, 
                 "confidence": round(d.confidence, 3), "label": d.label}
                for d in detections
            ]
        }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
        
        # Visualize
        if args.visualize:
            if args.input.lower().endswith(('.mp4', '.mov', '.avi')):
                cap = cv2.VideoCapture(args.input)
                ret, frame = cap.read()
                cap.release()
            else:
                frame = cv2.imread(args.input)
            
            for d in detections:
                cv2.rectangle(frame, (d.x, d.y), (d.x+d.w, d.y+d.h), (0, 255, 0), 2)
                cv2.putText(frame, f"{d.label} {d.confidence:.2f}", 
                           (d.x, d.y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            out_path = Path(args.input).stem + "_detected.jpg"
            cv2.imwrite(out_path, frame)
            print(f"Visualization saved to {out_path}")
    
    elif args.command == "train":
        trainer = YOLOTrainer(args.data_dir)
        if args.setup:
            trainer.setup_dataset()
        else:
            trainer.train(epochs=args.epochs)
    
    elif args.command == "annotate":
        trainer = YOLOTrainer(args.data_dir)
        trainer.setup_dataset()
        trainer.annotate_from_video(args.video, args.prefix, sample_frames=args.frames)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
