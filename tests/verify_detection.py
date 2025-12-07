"""
Verify face detection is working by testing with webcam screenshot
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.detection.retinaface import RetinaFace
from src.detection.utils import draw_detections

def main():
    print("\n" + "="*60)
    print("FACE DETECTION VERIFICATION")
    print("="*60)
    
    # Initialize detector
    model_path = Path("models") / "retinaface_resnet50.onnx"
    detector = RetinaFace(
        model_path=str(model_path),
        confidence_threshold=0.3,  # Lower threshold for testing
        nms_threshold=0.4
    )
    
    print("\n1. Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("[X] Failed to open webcam")
        return
    
    print("[OK] Webcam opened")
    print("\nInstructions:")
    print("  - Position your face in front of the camera")
    print("  - Press SPACE when ready to detect")
    print("  - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show preview
        cv2.imshow('Preview - Press SPACE to detect', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space bar
            print("\n2. Running detection...")
            detections = detector.detect(frame)
            
            print(f"[OK] Found {len(detections)} face(s)")
            
            if len(detections) > 0:
                for i, det in enumerate(detections):
                    bbox = det['bbox']
                    conf = det['confidence']
                    landmarks = det['landmarks']
                    print(f"   Face {i+1}: bbox={bbox}, confidence={conf:.3f}")
                    print(f"           landmarks={landmarks[:2]}...")  # Show first 2 landmarks
                
                # Draw and display
                result = draw_detections(frame.copy(), detections)
                cv2.imshow('Detection Result', result)
                print("\nResult displayed. Press any key to continue...")
                cv2.waitKey(0)
            else:
                print("[!] No faces detected. Try:")
                print("    - Moving closer to camera")
                print("    - Improving lighting")
                print("    - Adjusting face angle")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
