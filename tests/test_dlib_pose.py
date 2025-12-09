"""
Test improved head pose estimation with dlib 68-point landmarks
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.detection.retinaface import RetinaFace
from src.behavior.dlib_head_pose import DlibHeadPoseEstimator
from src.behavior.head_pose import estimate_attention_level

def main():
    print("\n" + "="*60)
    print("DLIB 68-POINT HEAD POSE ESTIMATION TEST")
    print("="*60)
    
    # Initialize models
    print("\n1. Initializing detector...")
    detector = RetinaFace(
        model_path="models/retinaface_resnet50.onnx",
        confidence_threshold=0.5
    )
    print("[OK] RetinaFace loaded")
    
    print("\n2. Initializing dlib pose estimator...")
    dlib_pose = DlibHeadPoseEstimator()
    print("[OK] Dlib 68-point pose estimator loaded")
    
    # Open webcam
    print("\n3. Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Set higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("[X] Failed to open webcam")
        return
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Webcam opened at {actual_width}x{actual_height}")
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("  - Dlib will detect 68 facial landmarks")
    print("  - Turn your head to test angle detection")
    print("  - Angles should now reach ±70-90° at extreme turns")
    print("  - Press 'q' to quit")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # Use dlib's own detector for landmarks (more accurate)
        landmarks_68 = dlib_pose.detect_landmarks(frame, bbox=None)
        
        if landmarks_68 is not None:
            # Calculate bbox from landmarks for display
            x1 = int(landmarks_68[:, 0].min())
            y1 = int(landmarks_68[:, 1].min())
            x2 = int(landmarks_68[:, 0].max())
            y2 = int(landmarks_68[:, 1].max())
            
            # Estimate pose from 68 landmarks
            yaw, pitch, roll = dlib_pose.estimate_pose(landmarks_68, (h, w))
            
            # Calculate attention
            attention_score, attention_level = estimate_attention_level(yaw, pitch, roll)
            
            # Draw bbox from landmarks
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Dlib Detection", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw 68 landmarks
            for i, (lx, ly) in enumerate(landmarks_68):
                # Color key landmarks differently
                if i in [30, 8, 36, 45, 48, 54]:  # Pose estimation points
                    cv2.circle(frame, (int(lx), int(ly)), 3, (0, 0, 255), -1)
                else:
                    cv2.circle(frame, (int(lx), int(ly)), 1, (0, 255, 255), -1)
            
            # Draw pose axes
            frame = dlib_pose.draw_pose_axes(
                frame, landmarks_68, yaw, pitch, roll, scale=100
            )
            
            # Create info panel
            panel_height = 180
            panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
            panel.fill(30)
            
            y_text = 30
            
            # Title
            cv2.putText(panel, "DLIB 68-POINT HEAD POSE", (20, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_text += 40
            
            # Angles
            yaw_color = (0, 255, 0) if abs(yaw) < 30 else (0, 255, 255) if abs(yaw) < 60 else (0, 0, 255)
            cv2.putText(panel, f"YAW (Left/Right):  {yaw:6.1f} deg", (20, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, yaw_color, 2)
            y_text += 35
            
            pitch_color = (0, 255, 0) if abs(pitch) < 20 else (0, 255, 255) if abs(pitch) < 40 else (0, 0, 255)
            cv2.putText(panel, f"PITCH (Up/Down):   {pitch:6.1f} deg", (20, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, pitch_color, 2)
            y_text += 35
            
            roll_color = (0, 255, 0) if abs(roll) < 20 else (0, 255, 255)
            cv2.putText(panel, f"ROLL (Tilt):       {roll:6.1f} deg", (20, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, roll_color, 2)
            
            # Attention info
            y_text = 30
            x_right = 450
            
            cv2.putText(panel, "ATTENTION", (x_right, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_text += 40
            
            att_color = (0, 255, 0) if attention_score >= 0.7 else (0, 255, 255) if attention_score >= 0.5 else (0, 0, 255)
            cv2.putText(panel, f"Score: {attention_score:.3f}", (x_right, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, att_color, 2)
            y_text += 35
            
            cv2.putText(panel, f"Level: {attention_level.upper()}", (x_right, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, att_color, 2)
            y_text += 35
            
            # Landmark count
            cv2.putText(panel, "68 landmarks detected", (x_right, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Stack
            combined = np.vstack([frame, panel])
            cv2.imshow('Dlib Head Pose', combined)
        else:
            cv2.putText(frame, "No face detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Dlib Head Pose', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nWith 68 landmarks, you should see:")
    print("  ✓ Yaw angles reaching ±70-90° at extreme turns")
    print("  ✓ More accurate pose tracking")
    print("  ✓ Better performance at large angles")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
