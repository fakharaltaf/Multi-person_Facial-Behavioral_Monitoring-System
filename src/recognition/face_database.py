"""
Face database for storing and matching face embeddings
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FaceDatabase:
    """
    Database for storing face embeddings and metadata
    """
    
    def __init__(self, db_path: str = "data/face_database"):
        """
        Initialize face database
        
        Args:
            db_path: Path to database directory
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_file = self.db_path / "embeddings.pkl"
        self.metadata_file = self.db_path / "metadata.json"
        
        # Load existing database
        self.embeddings = []  # List of embedding vectors
        self.metadata = []    # List of metadata dicts
        self.person_ids = []  # List of person IDs
        
        self._load_database()
        
        logger.info(f"Face database initialized: {len(self.embeddings)} faces")
    
    def _load_database(self):
        """Load database from disk"""
        try:
            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data.get('embeddings', [])
                    self.person_ids = data.get('person_ids', [])
            
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            
            logger.info(f"Loaded {len(self.embeddings)} embeddings from database")
        
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            self.embeddings = []
            self.metadata = []
            self.person_ids = []
    
    def _save_database(self):
        """Save database to disk"""
        try:
            # Save embeddings as pickle
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'person_ids': self.person_ids
                }, f)
            
            # Save metadata as JSON
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            logger.info(f"Saved {len(self.embeddings)} embeddings to database")
        
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def add_person(
        self,
        embedding: np.ndarray,
        person_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add a person to the database
        
        Args:
            embedding: Face embedding vector
            person_id: Unique identifier for person
            name: Person's name
            metadata: Additional metadata
        
        Returns:
            Index of added embedding
        """
        # Add embedding
        self.embeddings.append(embedding)
        self.person_ids.append(person_id)
        
        # Add metadata
        meta = {
            'person_id': person_id,
            'name': name or person_id,
            'added_at': datetime.now().isoformat(),
            'embedding_index': len(self.embeddings) - 1
        }
        
        if metadata:
            meta.update(metadata)
        
        self.metadata.append(meta)
        
        # Save to disk
        self._save_database()
        
        logger.info(f"Added person: {person_id} ({name})")
        
        return len(self.embeddings) - 1
    
    def find_match(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.6
    ) -> Tuple[Optional[str], float, Optional[Dict]]:
        """
        Find matching person in database
        
        Args:
            query_embedding: Query face embedding
            threshold: Minimum similarity threshold
        
        Returns:
            (person_id, similarity, metadata) or (None, 0.0, None)
        """
        if not self.embeddings:
            return None, 0.0, None
        
        # Compute similarities
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_embedding, emb)
            sim = (sim + 1) / 2  # Convert to [0, 1]
            similarities.append(sim)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        if best_sim >= threshold:
            person_id = self.person_ids[best_idx]
            metadata = self.metadata[best_idx]
            return person_id, best_sim, metadata
        else:
            return None, best_sim, None
    
    def get_person_info(self, person_id: str) -> Optional[Dict]:
        """Get metadata for a person"""
        for meta in self.metadata:
            if meta['person_id'] == person_id:
                return meta
        return None
    
    def list_persons(self) -> List[Dict]:
        """List all persons in database"""
        return self.metadata.copy()
    
    def remove_person(self, person_id: str) -> bool:
        """Remove a person from database"""
        indices_to_remove = []
        
        for i, pid in enumerate(self.person_ids):
            if pid == person_id:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            return False
        
        # Remove in reverse order to maintain indices
        for i in sorted(indices_to_remove, reverse=True):
            del self.embeddings[i]
            del self.person_ids[i]
            del self.metadata[i]
        
        # Update embedding indices in metadata
        for i, meta in enumerate(self.metadata):
            meta['embedding_index'] = i
        
        self._save_database()
        
        logger.info(f"Removed person: {person_id} ({len(indices_to_remove)} embeddings)")
        
        return True
    
    def clear_database(self):
        """Clear entire database"""
        self.embeddings = []
        self.metadata = []
        self.person_ids = []
        self._save_database()
        logger.info("Database cleared")
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        unique_persons = len(set(self.person_ids))
        
        return {
            'total_embeddings': len(self.embeddings),
            'unique_persons': unique_persons,
            'persons': list(set(self.person_ids))
        }
