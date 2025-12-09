"""
Enhanced tracker with behavioral analysis
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from ..tracking.tracker import MultiObjectTracker
from .behavior_track import BehaviorTrack

# Try to import dlib support
try:
    from .dlib_head_pose import DlibHeadPoseEstimator
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class BehaviorTracker(MultiObjectTracker):
    """
    Multi-object tracker with integrated behavioral analysis
    """
    
    def __init__(self, *args, use_dlib: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override tracks list to use BehaviorTrack
        self.tracks: List[BehaviorTrack] = []
        
        # Initialize dlib if requested and available
        self.use_dlib = use_dlib and DLIB_AVAILABLE
        self.dlib_estimator = None
        
        if self.use_dlib:
            try:
                self.dlib_estimator = DlibHeadPoseEstimator()
                logger.info("BehaviorTracker initialized with dlib 68-point landmarks")
            except Exception as e:
                logger.warning(f"Failed to initialize dlib: {e}. Falling back to 5-point landmarks")
                self.use_dlib = False
        else:
            if use_dlib and not DLIB_AVAILABLE:
                logger.warning("dlib requested but not available. Using 5-point landmarks")
            logger.info("BehaviorTracker initialized with 5-point landmarks")
    
    def update(
        self,
        detections: List[Dict],
        embeddings: Optional[List[np.ndarray]] = None,
        image_shape: Optional[tuple] = None,
        image: Optional[np.ndarray] = None
    ) -> List[BehaviorTrack]:
        """
        Update tracker with behavioral analysis
        
        Args:
            detections: List of detection dicts
            embeddings: Optional face embeddings
            image_shape: (height, width) for pose estimation
            image: Full image for dlib landmark detection
        
        Returns:
            List of active behavior tracks
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
        
        # Update matched tracks with behavioral analysis
        for det_idx, trk_idx in matched:
            detection = detections[det_idx]
            embedding = embeddings[det_idx] if embeddings else None
            
            self.tracks[trk_idx].update(
                bbox=detection['bbox'],
                embedding=embedding,
                landmarks=detection.get('landmarks'),
                confidence=detection.get('confidence', 0.0),
                image_shape=image_shape,
                image=image
            )
        
        # Mark unmatched tracks as missed
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].mark_missed()
        
        # Create new BehaviorTrack for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            embedding = embeddings[det_idx] if embeddings else None
            
            new_track = BehaviorTrack(
                bbox=detection['bbox'],
                embedding=embedding,
                landmarks=detection.get('landmarks'),
                confidence=detection.get('confidence', 0.0),
                use_dlib=self.use_dlib,
                dlib_estimator=self.dlib_estimator
            )
            
            # Initial behavioral analysis
            if self.use_dlib and self.dlib_estimator and image is not None:
                # Extract face region for dlib (expand bbox slightly for better detection)
                bbox = detection['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                h, w = image.shape[:2]
                
                # Expand bbox by 20%
                margin_x = int((x2 - x1) * 0.2)
                margin_y = int((y2 - y1) * 0.2)
                
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(w, x2 + margin_x)
                y2 = min(h, y2 + margin_y)
                
                face_region = image[y1:y2, x1:x2]
                
                if face_region.size > 0:
                    # Detect landmarks in face region (without bbox, let dlib detect)
                    new_track.landmarks_68 = self.dlib_estimator.detect_landmarks(face_region, bbox=None)
                    
                    if new_track.landmarks_68 is not None:
                        # Adjust landmarks back to full image coordinates
                        new_track.landmarks_68[:, 0] += x1
                        new_track.landmarks_68[:, 1] += y1
                        
                        if image_shape:
                            new_track._analyze_behavior_dlib(new_track.landmarks_68, image_shape)
            
            if new_track.landmarks_68 is None and detection.get('landmarks') and image_shape:
                new_track._analyze_behavior(detection['landmarks'], image_shape)
            
            self.tracks.append(new_track)
        
        # Remove invalid tracks
        self.tracks = [t for t in self.tracks if t.is_valid(self.max_age)]
        
        # Return only confirmed tracks
        active_tracks = [
            t for t in self.tracks 
            if t.state == 'confirmed' or (t.state == 'tentative' and t.hits >= 1)
        ]
        
        return active_tracks
    
    def get_engagement_statistics(self) -> Dict:
        """
        Get engagement statistics for all tracks
        
        Returns:
            Dictionary with engagement metrics
        """
        confirmed_tracks = [t for t in self.tracks if t.state == 'confirmed']
        
        if not confirmed_tracks:
            return {
                'total_people': 0,
                'engaged': 0,
                'moderate': 0,
                'disengaged': 0,
                'average_attention': 0.0,
                'drowsy_count': 0,
                'yawning_count': 0
            }
        
        engagement_counts = {'engaged': 0, 'moderate': 0, 'disengaged': 0}
        total_attention = 0.0
        drowsy_count = 0
        yawning_count = 0
        
        for track in confirmed_tracks:
            engagement = track.get_engagement_level()
            engagement_counts[engagement] += 1
            
            total_attention += track.attention_score
            
            if track.is_drowsy:
                drowsy_count += 1
            if track.is_yawning:
                yawning_count += 1
        
        avg_attention = total_attention / len(confirmed_tracks)
        
        return {
            'total_people': len(confirmed_tracks),
            'engaged': engagement_counts['engaged'],
            'moderate': engagement_counts['moderate'],
            'disengaged': engagement_counts['disengaged'],
            'average_attention': avg_attention,
            'drowsy_count': drowsy_count,
            'yawning_count': yawning_count
        }
