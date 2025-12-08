"""
Test complete behavioral analysis system
Detection → Recognition → Tracking → Behavioral Analysis
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
from src.behavior.behavior_tracker import BehaviorTracker
from src.behavior.visualization import create_behavior_visualization

def main():
    print("\n" + "="*60)
    print("BEHAVIORAL ANALYSIS SYSTEM TEST")
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
    
    tracker = BehaviorTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        embedding_threshold=0.5,
        use_embeddings=True
    )
    print("[OK] BehaviorTracker initialized")
    
    # Open webcam
    print("\n2. Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("[X] Failed to open webcam")
        return
    
    print("[OK] Webcam opened")
    print("\nInstructions:")
    print("  - Behavioral analysis runs automatically")
    print("  - Press 'r' to REGISTER current tracks")
    print("  - Press 'p' to toggle POSE axes")
    print("  - Press 'd' to toggle DETAILS")
    print("  - Press 's' to show STATISTICS")
    print("  - Press 'e' to show ENGAGEMENT report")
    print("  - Press 'c' to CLEAR all tracks")
    print("  - Press 'q' to QUIT")
    
    show_pose = True
    show_details = True
    frame_times = []
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # Detect faces
        detections = detector.detect(frame)
        
        # Extract embeddings for detected faces
        embeddings = []
        if len(detections) > 0:
            aligned_faces = align_faces_batch(frame, detections)
            embeddings = recognizer.extract_embeddings_batch(aligned_faces)
        
        # Update tracker with behavioral analysis
        tracks = tracker.update(detections, embeddings, image_shape=(h, w))
        
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
        
        # Visualize with behavioral overlay
        vis = create_behavior_visualization(
            frame,
            tracks,
            fps=avg_fps,
            show_pose=show_pose,
            show_details=show_details
        )
        
        cv2.imshow('Behavioral Analysis System', vis)
        
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
                print(f"  Engagement: {track.get_engagement_level()}")
                print(f"  Attention: {track.attention_score:.2f}")
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
        
        elif key == ord('p'):  # Toggle pose
            show_pose = not show_pose
            print(f"Pose axes: {'ON' if show_pose else 'OFF'}")
        
        elif key == ord('d'):  # Toggle details
            show_details = not show_details
            print(f"Detailed info: {'ON' if show_details else 'OFF'}")
        
        elif key == ord('s'):  # Show statistics
            print("\n--- TRACKER STATISTICS ---")
            stats = tracker.get_statistics()
            print(f"Frame count: {stats['frame_count']}")
            print(f"Total tracks: {stats['total_tracks']}")
            print(f"Confirmed tracks: {stats['confirmed_tracks']}")
            print(f"Average FPS: {avg_fps:.1f}")
            
            print("\n--- ACTIVE TRACKS ---")
            for track in tracks:
                print(f"Track {track.track_id} ({track.person_name}):")
                print(f"  State: {track.state}, Age: {track.age} frames")
                print(f"  Pose: Y={track.yaw:.1f} P={track.pitch:.1f} R={track.roll:.1f}")
                print(f"  Gaze: {track.gaze_direction}")
                print(f"  Attention: {track.attention_score:.2f} ({track.attention_level})")
                print(f"  Engagement: {track.get_engagement_level()}")
                print(f"  Blinks: {track.blink_count}, Yawns: {track.yawn_count}")
                if track.is_drowsy:
                    print(f"  [!] DROWSY (level: {track.drowsiness_level:.2f})")
        
        elif key == ord('e'):  # Engagement report
            print("\n--- ENGAGEMENT REPORT ---")
            eng_stats = tracker.get_engagement_statistics()
            print(f"Total people: {eng_stats['total_people']}")
            print(f"Engaged: {eng_stats['engaged']} ({eng_stats['engaged']/max(1,eng_stats['total_people'])*100:.1f}%)")
            print(f"Moderate: {eng_stats['moderate']} ({eng_stats['moderate']/max(1,eng_stats['total_people'])*100:.1f}%)")
            print(f"Disengaged: {eng_stats['disengaged']} ({eng_stats['disengaged']/max(1,eng_stats['total_people'])*100:.1f}%)")
            print(f"Average attention: {eng_stats['average_attention']:.2f}")
            if eng_stats['drowsy_count'] > 0:
                print(f"[!] Drowsy people: {eng_stats['drowsy_count']}")
            if eng_stats['yawning_count'] > 0:
                print(f"[!] Yawning people: {eng_stats['yawning_count']}")
        
        elif key == ord('c'):  # Clear tracks
            confirm = input("\nClear all tracks? (yes/no): ")
            if confirm.lower() == 'yes':
                tracker.reset()
                print("[OK] All tracks cleared")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final report
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    stats = tracker.get_statistics()
    print(f"Total frames processed: {stats['frame_count']}")
    print(f"Average FPS: {avg_fps:.1f}")
    
    eng_stats = tracker.get_engagement_statistics()
    if eng_stats['total_people'] > 0:
        print(f"\nFinal engagement statistics:")
        print(f"  Engaged: {eng_stats['engaged']}")
        print(f"  Moderate: {eng_stats['moderate']}")
        print(f"  Disengaged: {eng_stats['disengaged']}")
        print(f"  Average attention: {eng_stats['average_attention']:.2f}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
