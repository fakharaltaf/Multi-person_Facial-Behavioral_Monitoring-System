"""
Debug script to test yawn detection with 68-point landmarks
Shows real-time MAR values to help calibrate thresholds
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.detection.retinaface import RetinaFace
from src.behavior.dlib_head_pose import DlibHeadPoseEstimator

def compute_mar_68(mouth_points: np.ndarray) -> float:
    """Compute Mouth Aspect Ratio from 68-point mouth landmarks"""
    # Vertical distance (center of upper/lower lips)
    v1 = np.linalg.norm(mouth_points[13] - mouth_points[19])  # 61-67
    v2 = np.linalg.norm(mouth_points[14] - mouth_points[18])  # 62-66
    v3 = np.linalg.norm(mouth_points[15] - mouth_points[17])  # 63-65
    # Horizontal distance
    h = np.linalg.norm(mouth_points[0] - mouth_points[6])  # 48-54
    # MAR
    return (v1 + v2 + v3) / (3.0 * h + 1e-6)

def main():
    print("\n" + "="*60)
    print("YAWN DETECTION DEBUG - 68-Point Landmarks")
    print("="*60)
    
    # Initialize models
    print("\n1. Initializing models...")
    detector = RetinaFace(
        model_path="models/retinaface_resnet50.onnx",
        confidence_threshold=0.5
    )
    print("[OK] RetinaFace loaded")
    
    dlib_pose = DlibHeadPoseEstimator()
    print("[OK] Dlib loaded")
    
    # Open webcam
    print("\n2. Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("[X] Failed to open webcam")
        return
    
    print("[OK] Webcam opened at 1280x720")
    print("\nInstructions:")
    print("  - Keep your face neutral to see baseline MAR")
    print("  - Open your mouth wide (simulate yawn) to see MAR increase")
    print("  - MAR > 0.6-0.7 typically indicates yawning")
    print("  - Press 'q' to QUIT")
    
    mar_history = []
    max_mar = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # Detect faces
        detections = detector.detect(frame)
        
        if len(detections) > 0:
            # Use first detection
            detection = detections[0]
            bbox = detection['bbox']
            
            # Expand bbox for better dlib detection
            x1, y1, x2, y2 = map(int, bbox)
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
            
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size > 0:
                # Detect 68-point landmarks
                landmarks_68 = dlib_pose.detect_landmarks(face_region, bbox=None)
                
                if landmarks_68 is not None:
                    # Adjust landmarks back to full image coordinates
                    landmarks_68[:, 0] += x1
                    landmarks_68[:, 1] += y1
                    
                    # Extract mouth landmarks (48-67)
                    mouth = landmarks_68[48:68]
                    
                    # Compute MAR
                    mar = compute_mar_68(mouth)
                    mar_history.append(mar)
                    if len(mar_history) > 30:
                        mar_history.pop(0)
                    
                    avg_mar = sum(mar_history) / len(mar_history)
                    max_mar = max(max_mar, mar)
                    
                    # Draw mouth landmarks
                    for i, (x, y) in enumerate(mouth):
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    
                    # Display MAR info
                    info_y = 30
                    cv2.putText(frame, f"MAR: {mar:.3f}", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    info_y += 40
                    
                    cv2.putText(frame, f"Avg MAR: {avg_mar:.3f}", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    info_y += 35
                    
                    cv2.putText(frame, f"Max MAR: {max_mar:.3f}", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    info_y += 40
                    
                    # Show yawn status with different thresholds
                    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
                    for threshold in thresholds:
                        is_yawn = mar > threshold
                        color = (0, 0, 255) if is_yawn else (128, 128, 128)
                        text = f"Threshold {threshold}: {'YAWN' if is_yawn else 'Normal'}"
                        cv2.putText(frame, text, (10, info_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        info_y += 30
                    
                    # Visual MAR bar
                    bar_x = w - 150
                    bar_y = 50
                    bar_height = 400
                    bar_width = 30
                    
                    # Background bar
                    cv2.rectangle(frame, (bar_x, bar_y), 
                                (bar_x + bar_width, bar_y + bar_height), 
                                (100, 100, 100), -1)
                    
                    # Current MAR level (capped at 1.5 for display)
                    mar_normalized = min(mar / 1.5, 1.0)
                    fill_height = int(bar_height * mar_normalized)
                    
                    # Color based on threshold
                    if mar > 0.7:
                        bar_color = (0, 0, 255)  # Red - likely yawn
                    elif mar > 0.5:
                        bar_color = (0, 165, 255)  # Orange - possible yawn
                    else:
                        bar_color = (0, 255, 0)  # Green - normal
                    
                    cv2.rectangle(frame, 
                                (bar_x, bar_y + bar_height - fill_height), 
                                (bar_x + bar_width, bar_y + bar_height), 
                                bar_color, -1)
                    
                    # Threshold lines
                    for threshold in [0.5, 0.6, 0.7, 0.8]:
                        threshold_y = bar_y + bar_height - int(bar_height * (threshold / 1.5))
                        cv2.line(frame, (bar_x - 5, threshold_y), 
                               (bar_x + bar_width + 5, threshold_y), 
                               (255, 255, 255), 1)
                        cv2.putText(frame, f"{threshold}", (bar_x + bar_width + 10, threshold_y + 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Yawn Detection Debug', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[*] Session complete")
    print(f"[*] Maximum MAR observed: {max_mar:.3f}")
    print(f"[*] Recommended threshold: ~0.6-0.7 for sensitive detection")

if __name__ == "__main__":
    main()
