"""
Debug face recognition similarities
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.detection.retinaface import RetinaFace
from src.recognition.alignment import align_faces_batch
from src.recognition.arcface import ArcFace
from src.recognition.face_database import FaceDatabase

def main():
    print("\n" + "="*60)
    print("FACE RECOGNITION SIMILARITY DEBUG")
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
    
    database = FaceDatabase()
    print(f"[OK] Database loaded: {database.get_statistics()}")
    
    # Open webcam
    print("\n2. Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("[X] Failed to open webcam")
        return
    
    print("[OK] Webcam opened")
    print("\nPress 's' to check SIMILARITIES for detected faces")
    print("Press 'q' to QUIT\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        detections = detector.detect(frame)
        
        # Draw detections
        for idx, det in enumerate(detections):
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {idx+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Detected: {len(detections)} faces", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow('Recognition Debug', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n" + "="*60)
            print(f"Checking similarities for {len(detections)} detected face(s)...")
            print("="*60)
            
            if len(detections) == 0:
                print("No faces detected!")
                continue
            
            # Extract embeddings
            aligned_faces = align_faces_batch(frame, detections)
            embeddings = recognizer.extract_embeddings_batch(aligned_faces)
            
            # Check each face against database
            for idx, embedding in enumerate(embeddings):
                print(f"\nFace {idx+1}:")
                
                # Normalize embedding
                emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
                print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
                
                # Check against all database entries
                if len(database.embeddings) == 0:
                    print("  [!] Database is empty - no comparisons possible")
                    continue
                
                print(f"  Comparing against {len(database.embeddings)} database entry(ies):")
                
                for db_idx, (db_emb, person_id, meta) in enumerate(zip(
                    database.embeddings, database.person_ids, database.metadata
                )):
                    # Normalize database embedding
                    db_emb_norm = db_emb / (np.linalg.norm(db_emb) + 1e-8)
                    
                    # Cosine similarity
                    cos_sim = np.dot(emb_norm, db_emb_norm)
                    
                    # Convert to [0, 1]
                    sim_01 = (cos_sim + 1.0) / 2.0
                    
                    person_name = meta.get('name', person_id)
                    
                    match_status = "✓ MATCH" if sim_01 >= 0.75 else "✗ No match"
                    
                    print(f"    [{db_idx+1}] {person_name} (ID: {person_id})")
                    print(f"        Cosine similarity: {cos_sim:.4f} (raw)")
                    print(f"        Scaled similarity: {sim_01:.4f} (0-1 range)")
                    print(f"        Status: {match_status} (threshold: 0.75)")
                
                # Also show what find_match returns
                person_id, similarity, metadata = database.find_match(embedding, threshold=0.75)
                if person_id:
                    print(f"\n  find_match() result: {metadata['name']} with similarity {similarity:.4f}")
                else:
                    print(f"\n  find_match() result: No match (best similarity: {similarity:.4f})")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
