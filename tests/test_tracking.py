"""
Test multi-object tracking with face recognition
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import time
from src.detection.retinaface import RetinaFace
from src.recognition.alignment import align_faces_batch
from src.recognition.arcface import ArcFace
from src.recognition.face_database import FaceDatabase
from src.tracking.tracker import MultiObjectTracker
from src.tracking.visualization import create_tracking_visualization

def main():
    print("\n" + "="*60)
    print("MULTI-OBJECT TRACKING TEST")
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
    
    # Initialize database and tracker
    database = FaceDatabase()
    print(f"[OK] Database loaded: {database.get_statistics()}")
    
    tracker = MultiObjectTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        embedding_threshold=0.5,
        use_embeddings=True
    )
    print("[OK] Tracker initialized")
    
    # Open webcam
    print("\n2. Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("[X] Failed to open webcam")
        return
    
    print("[OK] Webcam opened")
    print("\nInstructions:")
    print("  - Tracking runs automatically")
    print("  - Press 'r' to REGISTER current tracks")
    print("  - Press 't' to toggle TRAJECTORY display")
    print("  - Press 'g' to toggle GRID overlay")
    print("  - Press 's' to show STATISTICS")
    print("  - Press 'c' to CLEAR all tracks")
    print("  - Press 'q' to QUIT")
    
    show_trajectory = True
    show_grid = False
    frame_times = []
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        detections = detector.detect(frame)
        
        # Extract embeddings for detected faces
        embeddings = []
        if len(detections) > 0:
            aligned_faces = align_faces_batch(frame, detections)
            embeddings = recognizer.extract_embeddings_batch(aligned_faces)
        
        # Update tracker
        tracks = tracker.update(detections, embeddings)
        
        # Match tracks with database
        for track in tracks:
            if track.embedding is not None and track.person_id is None:
                person_id, similarity, metadata = database.find_match(
                    track.embedding,
                    threshold=0.6
                )
                if person_id:
                    track.set_identity(
                        person_id,
                        metadata['name'],
                        similarity
                    )
        
        # Calculate FPS
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
        
        # Visualize
        vis = create_tracking_visualization(
            frame,
            tracks,
            fps=avg_fps,
            show_trajectory=show_trajectory,
            show_grid=show_grid
        )
        
        cv2.imshow('Multi-Object Tracking', vis)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('r'):  # Register tracks
            print("\n--- REGISTRATION MODE ---")
            if len(tracks) == 0:
                print("[!] No active tracks to register")
                continue
            
            print(f"Found {len(tracks)} active track(s)")
            for track in tracks:
                if track.embedding is None:
                    print(f"  Track {track.track_id}: No embedding available")
                    continue
                
                if track.person_id is not None:
                    print(f"  Track {track.track_id}: Already registered as {track.person_name}")
                    continue
                
                print(f"\nRegister Track {track.track_id}?")
                response = input("  Enter name (or press Enter to skip): ").strip()
                
                if response:
                    person_id = f"person_{track.track_id:03d}"
                    database.add_person(
                        track.get_average_embedding(),
                        person_id,
                        response
                    )
                    track.set_identity(person_id, response, 1.0)
                    print(f"  [OK] Registered as {response}")
        
        elif key == ord('t'):  # Toggle trajectory
            show_trajectory = not show_trajectory
            print(f"Trajectory display: {'ON' if show_trajectory else 'OFF'}")
        
        elif key == ord('g'):  # Toggle grid
            show_grid = not show_grid
            print(f"Grid overlay: {'ON' if show_grid else 'OFF'}")
        
        elif key == ord('s'):  # Show statistics
            print("\n--- TRACKER STATISTICS ---")
            stats = tracker.get_statistics()
            print(f"Frame count: {stats['frame_count']}")
            print(f"Total tracks: {stats['total_tracks']}")
            print(f"Confirmed tracks: {stats['confirmed_tracks']}")
            print(f"Tentative tracks: {stats['tentative_tracks']}")
            print(f"Average FPS: {avg_fps:.1f}")
            
            print("\n--- ACTIVE TRACKS ---")
            for track in tracks:
                print(f"Track {track.track_id}:")
                print(f"  Person: {track.person_name}")
                print(f"  State: {track.state}")
                print(f"  Age: {track.age} frames")
                print(f"  Hits: {track.hits}")
                print(f"  Identity conf: {track.identity_confidence:.3f}")
            
            print("\n--- DATABASE ---")
            db_stats = database.get_statistics()
            print(f"Total embeddings: {db_stats['total_embeddings']}")
            print(f"Unique persons: {db_stats['unique_persons']}")
        
        elif key == ord('c'):  # Clear tracks
            confirm = input("\nClear all tracks? (yes/no): ")
            if confirm.lower() == 'yes':
                tracker.reset()
                print("[OK] All tracks cleared")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    stats = tracker.get_statistics()
    print(f"Total frames processed: {stats['frame_count']}")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Total tracks created: {tracker.tracks[-1].track_id if tracker.tracks else 0}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
