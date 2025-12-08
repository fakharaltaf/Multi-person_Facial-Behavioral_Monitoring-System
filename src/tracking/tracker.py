"""
Multi-object tracker using Hungarian algorithm for assignment
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
import logging

from .track import Track, compute_iou, compute_embedding_distance, compute_center_distance

logger = logging.getLogger(__name__)


class MultiObjectTracker:
    """
    Multi-object tracker for face tracking
    Uses appearance (embeddings) + motion (IoU, center distance) for association
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        embedding_threshold: float = 0.5,
        use_embeddings: bool = True
    ):
        """
        Initialize tracker
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: Minimum IoU for matching
            embedding_threshold: Maximum embedding distance for matching
            use_embeddings: Whether to use face embeddings for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.embedding_threshold = embedding_threshold
        self.use_embeddings = use_embeddings
        
        self.tracks: List[Track] = []
        self.frame_count = 0
        
        logger.info(f"Tracker initialized: max_age={max_age}, min_hits={min_hits}")
    
    def update(
        self,
        detections: List[Dict],
        embeddings: Optional[List[np.ndarray]] = None
    ) -> List[Track]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with 'bbox', 'landmarks', 'confidence'
            embeddings: Optional list of face embeddings (same length as detections)
        
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Predict next state for all tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        if len(detections) > 0 and len(self.tracks) > 0:
            matched, unmatched_dets, unmatched_trks = self._match_detections_to_tracks(
                detections, embeddings
            )
        else:
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.tracks)))
        
        # Update matched tracks
        for det_idx, trk_idx in matched:
            detection = detections[det_idx]
            embedding = embeddings[det_idx] if embeddings else None
            
            self.tracks[trk_idx].update(
                bbox=detection['bbox'],
                embedding=embedding,
                landmarks=detection.get('landmarks'),
                confidence=detection.get('confidence', 0.0)
            )
        
        # Mark unmatched tracks as missed
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            embedding = embeddings[det_idx] if embeddings else None
            
            new_track = Track(
                bbox=detection['bbox'],
                embedding=embedding,
                landmarks=detection.get('landmarks'),
                confidence=detection.get('confidence', 0.0)
            )
            self.tracks.append(new_track)
        
        # Remove invalid tracks
        self.tracks = [t for t in self.tracks if t.is_valid(self.max_age)]
        
        # Return only confirmed tracks
        active_tracks = [
            t for t in self.tracks 
            if t.state == 'confirmed' or (t.state == 'tentative' and t.hits >= 1)
        ]
        
        return active_tracks
    
    def _match_detections_to_tracks(
        self,
        detections: List[Dict],
        embeddings: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using Hungarian algorithm
        
        Returns:
            (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Build cost matrix
        cost_matrix = self._build_cost_matrix(detections, embeddings)
        
        # Hungarian algorithm
        det_indices, trk_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by threshold
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.tracks)))
        
        for det_idx, trk_idx in zip(det_indices, trk_indices):
            cost = cost_matrix[det_idx, trk_idx]
            
            # Cost threshold (lower is better)
            if cost < 1.0:  # Adjust threshold as needed
                matched.append((det_idx, trk_idx))
                unmatched_dets.remove(det_idx)
                unmatched_trks.remove(trk_idx)
        
        return matched, unmatched_dets, unmatched_trks
    
    def _build_cost_matrix(
        self,
        detections: List[Dict],
        embeddings: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Build cost matrix for matching
        Combines IoU, center distance, and embedding distance
        
        Returns:
            Cost matrix (detections x tracks)
        """
        num_dets = len(detections)
        num_trks = len(self.tracks)
        
        cost_matrix = np.zeros((num_dets, num_trks))
        
        for d in range(num_dets):
            det = detections[d]
            det_bbox = det['bbox']
            det_emb = embeddings[d] if embeddings else None
            
            for t in range(num_trks):
                track = self.tracks[t]
                
                # IoU cost (higher IoU = lower cost)
                iou = compute_iou(det_bbox, track.bbox)
                iou_cost = 1.0 - iou
                
                # Center distance cost
                center_dist = compute_center_distance(det_bbox, track.bbox)
                
                # Embedding distance cost
                if self.use_embeddings and det_emb is not None:
                    track_emb = track.get_average_embedding()
                    emb_dist = compute_embedding_distance(det_emb, track_emb)
                else:
                    emb_dist = 0.0
                
                # Combined cost (weighted sum)
                # Weights: IoU=0.4, Center=0.2, Embedding=0.4
                if self.use_embeddings and det_emb is not None:
                    cost = 0.3 * iou_cost + 0.2 * center_dist + 0.5 * emb_dist
                else:
                    cost = 0.6 * iou_cost + 0.4 * center_dist
                
                cost_matrix[d, t] = cost
        
        return cost_matrix
    
    def get_tracks(self, confirmed_only: bool = True) -> List[Track]:
        """
        Get all active tracks
        
        Args:
            confirmed_only: Return only confirmed tracks
        
        Returns:
            List of tracks
        """
        if confirmed_only:
            return [t for t in self.tracks if t.state == 'confirmed']
        else:
            return [t for t in self.tracks if t.state != 'deleted']
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID"""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def reset(self):
        """Reset tracker"""
        self.tracks = []
        self.frame_count = 0
        Track._next_id = 1
        logger.info("Tracker reset")
    
    def get_statistics(self) -> Dict:
        """Get tracker statistics"""
        total_tracks = len(self.tracks)
        confirmed = len([t for t in self.tracks if t.state == 'confirmed'])
        tentative = len([t for t in self.tracks if t.state == 'tentative'])
        
        return {
            'frame_count': self.frame_count,
            'total_tracks': total_tracks,
            'confirmed_tracks': confirmed,
            'tentative_tracks': tentative
        }
