"""
Test face recognition pipeline
Detection → Alignment → Embedding Extraction → Matching
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.detection.retinaface import RetinaFace
from src.recognition.alignment import align_face, visualize_landmarks
from src.recognition.arcface import ArcFace
from src.recognition.face_database import FaceDatabase

def main():
    print("\n" + "="*60)
    print("FACE RECOGNITION TEST")
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
    
    # Initialize database
    database = FaceDatabase()
    print(f"[OK] Database loaded: {database.get_statistics()}")
    
    # Open webcam
    print("\n2. Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("[X] Failed to open webcam")
        return
    
    print("[OK] Webcam opened")
    print("\nInstructions:")
    print("  - Press 'r' to REGISTER a new face")
    print("  - Press SPACE to RECOGNIZE faces")
    print("  - Press 'l' to LIST registered persons")
    print("  - Press 'c' to CLEAR database")
    print("  - Press 'q' to QUIT")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show preview
        display = frame.copy()
        cv2.putText(display, "Press 'r' to register | SPACE to recognize | 'q' to quit",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Face Recognition', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('r'):  # Register
            print("\n--- REGISTRATION MODE ---")
            print("Detecting faces...")
            
            detections = detector.detect(frame)
            
            if len(detections) == 0:
                print("[!] No faces detected. Try again.")
                continue
            
            if len(detections) > 1:
                print(f"[!] Multiple faces detected ({len(detections)}). Please ensure only one person.")
                continue
            
            det = detections[0]
            print(f"[OK] Detected 1 face (confidence: {det['confidence']:.3f})")
            
            # Align face
            try:
                aligned = align_face(frame, det['landmarks'])
                print("[OK] Face aligned")
                
                # Extract embedding
                embedding = recognizer.extract_embedding(aligned)
                print(f"[OK] Embedding extracted: shape={embedding.shape}")
                
                # Get person info
                person_id = input("Enter person ID (e.g., person_001): ").strip()
                if not person_id:
                    print("[!] Registration cancelled")
                    continue
                
                name = input("Enter name (optional): ").strip() or person_id
                
                # Add to database
                idx = database.add_person(embedding, person_id, name)
                print(f"[OK] Registered: {name} (ID: {person_id})")
                
                # Show aligned face
                cv2.imshow('Registered Face', aligned)
                cv2.waitKey(1000)
                
            except Exception as e:
                print(f"[X] Registration failed: {e}")
        
        elif key == ord(' '):  # Recognize
            print("\n--- RECOGNITION MODE ---")
            print("Detecting faces...")
            
            detections = detector.detect(frame)
            print(f"[OK] Detected {len(detections)} face(s)")
            
            if len(detections) == 0:
                print("[!] No faces detected")
                continue
            
            result_frame = frame.copy()
            
            for i, det in enumerate(detections):
                try:
                    # Align face
                    aligned = align_face(frame, det['landmarks'])
                    
                    # Extract embedding
                    embedding = recognizer.extract_embedding(aligned)
                    
                    # Match with database
                    person_id, similarity, metadata = database.find_match(
                        embedding,
                        threshold=0.6
                    )
                    
                    # Draw result
                    bbox = det['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    if person_id:
                        name = metadata['name']
                        label = f"{name} ({similarity:.2f})"
                        color = (0, 255, 0)
                        print(f"  Face {i+1}: {name} (similarity: {similarity:.3f})")
                    else:
                        label = f"Unknown ({similarity:.2f})"
                        color = (0, 0, 255)
                        print(f"  Face {i+1}: Unknown (best match: {similarity:.3f})")
                    
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(result_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                except Exception as e:
                    print(f"  Face {i+1}: Error - {e}")
            
            cv2.imshow('Recognition Result', result_frame)
            print("\nPress any key to continue...")
            cv2.waitKey(0)
        
        elif key == ord('l'):  # List persons
            print("\n--- DATABASE CONTENTS ---")
            stats = database.get_statistics()
            print(f"Total embeddings: {stats['total_embeddings']}")
            print(f"Unique persons: {stats['unique_persons']}")
            print("\nRegistered persons:")
            for person in database.list_persons():
                print(f"  - {person['name']} (ID: {person['person_id']})")
        
        elif key == ord('c'):  # Clear database
            confirm = input("\nAre you sure you want to clear the database? (yes/no): ")
            if confirm.lower() == 'yes':
                database.clear_database()
                print("[OK] Database cleared")
            else:
                print("[!] Cancelled")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
