"""
Debug script to see what detections are being generated
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.detection.retinaface import RetinaFace

def main():
    print("\n" + "="*60)
    print("DETECTION DEBUG TEST")
    print("="*60)
    
    # Initialize detector
    print("\n1. Initializing detector...")
    detector = RetinaFace(
        model_path="models/retinaface_resnet50.onnx",
        confidence_threshold=0.5,
        nms_threshold=0.25
    )
    print("[OK] RetinaFace loaded")
    
    # Open webcam
    print("\n2. Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("[X] Failed to open webcam")
        return
    
    print("[OK] Webcam opened")
    print("\nPress 'q' to quit, 'd' to print detection details\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect faces
        detections = detector.detect(frame)
        
        # Draw all detections with detailed info
        for idx, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['confidence']
            x1, y1, x2, y2 = bbox
            
            # Calculate area
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Draw bbox
            color = (0, 255, 0) if idx == 0 else (0, 0, 255)  # Green for first, red for others
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with details
            label = f"#{idx+1} conf:{conf:.2f} size:{width}x{height}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw landmarks
            for lm in det['landmarks']:
                cv2.circle(frame, tuple(lm), 2, color, -1)
        
        # Show count
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow('Detection Debug', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            print(f"\n--- Frame {frame_count} ---")
            print(f"Total detections: {len(detections)}")
            for idx, det in enumerate(detections):
                bbox = det['bbox']
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                area = width * height
                print(f"  Detection {idx+1}:")
                print(f"    Bbox: [{x1}, {y1}, {x2}, {y2}]")
                print(f"    Size: {width} x {height} (area: {area})")
                print(f"    Confidence: {det['confidence']:.4f}")
                
                # Calculate IoU between all pairs
                if len(detections) > 1:
                    for jdx, other in enumerate(detections):
                        if idx >= jdx:
                            continue
                        other_bbox = other['bbox']
                        ox1, oy1, ox2, oy2 = other_bbox
                        
                        # Calculate IoU
                        xx1 = max(x1, ox1)
                        yy1 = max(y1, oy1)
                        xx2 = min(x2, ox2)
                        yy2 = min(y2, oy2)
                        
                        w = max(0, xx2 - xx1)
                        h = max(0, yy2 - yy1)
                        inter = w * h
                        
                        area_other = (ox2 - ox1) * (oy2 - oy1)
                        iou = inter / (area + area_other - inter) if (area + area_other - inter) > 0 else 0
                        
                        print(f"    IoU with detection {jdx+1}: {iou:.4f}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
