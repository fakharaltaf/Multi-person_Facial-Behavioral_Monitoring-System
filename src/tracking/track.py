"""
Multi-object tracking for face tracking across frames
Uses combination of visual embeddings and spatial information
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class Track:
    """
    Represents a single tracked person
    """
    
    _next_id = 1  # Class variable for auto-incrementing IDs
    
    def __init__(
        self,
        track_id: Optional[int] = None,
        bbox: List[int] = None,
        embedding: np.ndarray = None,
        landmarks: List[List[int]] = None,
        confidence: float = 0.0
    ):
        """
        Initialize a track
        
        Args:
            track_id: Unique track ID (auto-assigned if None)
            bbox: Bounding box [x1, y1, x2, y2]
            embedding: Face embedding vector
            landmarks: 5-point facial landmarks
            confidence: Detection confidence
        """
        self.track_id = track_id if track_id is not None else Track._next_id
        if track_id is None:
            Track._next_id += 1
        
        self.bbox = bbox
        self.embedding = embedding
        self.landmarks = landmarks
        self.confidence = confidence
        
        # Track state
        self.age = 0  # Frames since creation
        self.hits = 1  # Number of successful detections
        self.time_since_update = 0  # Frames since last update
        self.state = 'tentative'  # 'tentative', 'confirmed', 'deleted'
        
        # History for trajectory and behavior analysis
        self.bbox_history = deque(maxlen=30)  # Last 30 frames
        self.embedding_history = deque(maxlen=10)  # Last 10 embeddings
        self.position_history = deque(maxlen=60)  # Last 60 positions (2 sec @ 30fps)
        
        # Store initial data
        if bbox:
            self.bbox_history.append(bbox)
            center = self._bbox_to_center(bbox)
            self.position_history.append(center)
        
        if embedding is not None:
            self.embedding_history.append(embedding)
        
        # Velocity estimation
        self.velocity = np.array([0.0, 0.0])
        
        # Identity information (from face database)
        self.person_id = None
        self.person_name = "Unknown"
        self.identity_confidence = 0.0
    
    def update(
        self,
        bbox: List[int],
        embedding: Optional[np.ndarray] = None,
        landmarks: Optional[List[List[int]]] = None,
        confidence: float = 0.0
    ):
        """
        Update track with new detection
        
        Args:
            bbox: New bounding box
            embedding: New face embedding
            landmarks: New landmarks
            confidence: Detection confidence
        """
        self.bbox = bbox
        self.landmarks = landmarks
        self.confidence = confidence
        
        # Update history
        self.bbox_history.append(bbox)
        center = self._bbox_to_center(bbox)
        self.position_history.append(center)
        
        if embedding is not None:
            self.embedding_history.append(embedding)
            self.embedding = embedding
        
        # Update velocity
        if len(self.position_history) >= 2:
            self.velocity = np.array(center) - np.array(self.position_history[-2])
        
        # Update state
        self.hits += 1
        self.time_since_update = 0
        
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
    
    def predict(self):
        """
        Predict next position using velocity
        """
        self.age += 1
        self.time_since_update += 1
        
        if self.bbox and len(self.velocity) == 2:
            # Simple constant velocity prediction
            center = self._bbox_to_center(self.bbox)
            predicted_center = np.array(center) + self.velocity
            
            # Update bbox with predicted center
            w = self.bbox[2] - self.bbox[0]
            h = self.bbox[3] - self.bbox[1]
            
            self.bbox = [
                int(predicted_center[0] - w/2),
                int(predicted_center[1] - h/2),
                int(predicted_center[0] + w/2),
                int(predicted_center[1] + h/2)
            ]
    
    def mark_missed(self):
        """Mark track as missed (no detection this frame)"""
        self.time_since_update += 1
        if self.state == 'tentative':
            self.state = 'deleted'
    
    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Get average of recent embeddings"""
        if not self.embedding_history:
            return None
        
        embeddings = np.array(list(self.embedding_history))
        avg_embedding = np.mean(embeddings, axis=0)
        # Normalize
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        return avg_embedding
    
    def set_identity(self, person_id: str, person_name: str, confidence: float):
        """Set track identity from face database"""
        self.person_id = person_id
        self.person_name = person_name
        self.identity_confidence = confidence
    
    def _bbox_to_center(self, bbox: List[int]) -> Tuple[float, float]:
        """Convert bbox to center point"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_center(self) -> Optional[Tuple[float, float]]:
        """Get current center position"""
        if self.bbox:
            return self._bbox_to_center(self.bbox)
        return None
    
    def is_valid(self, max_age: int = 30) -> bool:
        """Check if track is still valid"""
        if self.state == 'deleted':
            return False
        if self.time_since_update > max_age:
            return False
        return True
    
    def to_dict(self) -> Dict:
        """Convert track to dictionary for serialization"""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'landmarks': self.landmarks,
            'confidence': self.confidence,
            'person_id': self.person_id,
            'person_name': self.person_name,
            'identity_confidence': self.identity_confidence,
            'age': self.age,
            'hits': self.hits,
            'state': self.state,
            'velocity': self.velocity.tolist() if self.velocity is not None else [0, 0]
        }


def compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Compute IoU between two bounding boxes
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU score (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_embedding_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine distance between embeddings
    
    Args:
        emb1: First embedding
        emb2: Second embedding
    
    Returns:
        Distance (0-2, lower is more similar)
    """
    if emb1 is None or emb2 is None:
        return 2.0  # Maximum distance
    
    # Cosine similarity
    similarity = np.dot(emb1, emb2)
    # Convert to distance (0-2 range)
    distance = 1.0 - similarity
    
    return distance


def compute_center_distance(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Compute normalized Euclidean distance between bbox centers
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        Normalized distance
    """
    c1_x = (bbox1[0] + bbox1[2]) / 2
    c1_y = (bbox1[1] + bbox1[3]) / 2
    c2_x = (bbox2[0] + bbox2[2]) / 2
    c2_y = (bbox2[1] + bbox2[3]) / 2
    
    # Diagonal of bbox1 for normalization
    diag = np.sqrt((bbox1[2] - bbox1[0])**2 + (bbox1[3] - bbox1[1])**2)
    
    dist = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
    
    return dist / (diag + 1e-6)
