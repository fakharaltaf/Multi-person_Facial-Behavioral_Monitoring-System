"""
Simple test to verify yawn detection is working with real-time MAR display
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.detection.retinaface import RetinaFace
from src.behavior.behavior_tracker import BehaviorTracker
from src.recognition.arcface import ArcFace
from src.recognition.alignment import align_faces_batch

def main():
    print("\n" + "="*60)
    print("YAWN DETECTION TEST")
    print("="*60)
    
    # Initialize models
    print("\n1. Initializing models...")
    detector = RetinaFace(
        model_path="models/retinaface_resnet50.onnx",
        confidence_threshold=0.5
    )
    print("[OK] RetinaFace loaded")
    
    recognizer = ArcFace(
        model_path="models/arcface_resnet100.onnx"
    )
    print("[OK] ArcFace loaded")
    
    tracker = BehaviorTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        embedding_threshold=0.5,
        use_embeddings=True,
        use_dlib=True
    )
    print("[OK] BehaviorTracker initialized with dlib")
    
    # Open webcam
    print("\n2. Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("[X] Failed to open webcam")
        return
    
    print("[OK] Webcam opened")
    print("\nInstructions:")
    print("  - Watch the MAR value on screen")
    print("  - Open your mouth wide to simulate a yawn")
    print("  - MAR should go above 0.6 to trigger yawn detection")
    print("  - Press 'q' to QUIT")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # Detect faces
        detections = detector.detect(frame)
        embeddings = []
        
        if len(detections) > 0:
            aligned_faces = align_faces_batch(frame, detections)
            embeddings = recognizer.extract_embeddings_batch(aligned_faces)
        
        # Update tracker
        tracks = tracker.update(detections, embeddings, image_shape=(h, w), image=frame)
        
        # Display info for each track
        for track in tracks:
            if track.bbox is None:
                continue
            
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            
            # Draw bounding box
            color = (0, 255, 0) if not track.is_yawning else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display MAR and yawn info
            info_y = y2 + 30
            cv2.putText(frame, f"MAR: {track.mar:.3f}", (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            info_y += 30
            
            cv2.putText(frame, f"Threshold: 0.600", (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 30
            
            if track.is_yawning:
                cv2.putText(frame, "YAWNING!", (x1, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                info_y += 30
            
            cv2.putText(frame, f"Total Yawns: {track.yawn_count}", (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display landmarks if available
            if hasattr(track, 'landmarks_68') and track.landmarks_68 is not None:
                # Draw mouth landmarks only
                mouth = track.landmarks_68[48:68]
                for i, (x, y) in enumerate(mouth):
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        cv2.imshow('Yawn Detection Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n[*] Test complete")
    for track in tracks:
        print(f"Track {track.track_id}: {track.yawn_count} yawns detected")

if __name__ == "__main__":
    main()
