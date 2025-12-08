"""
Enhanced track with behavioral analysis
"""

import numpy as np
from typing import List, Dict, Optional
from collections import deque

from ..tracking.track import Track
from .head_pose import HeadPoseEstimator, get_gaze_direction, estimate_attention_level
from .eye_tracking import (
    BlinkDetector, 
    YawnDetector,
    compute_ear_from_5_landmarks,
    compute_mouth_aspect_ratio
)


class BehaviorTrack(Track):
    """
    Extended Track with behavioral analysis capabilities
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Behavioral analyzers
        self.pose_estimator = HeadPoseEstimator()
        self.blink_detector = BlinkDetector()
        self.yawn_detector = YawnDetector()
        
        # Current behavioral state
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.gaze_direction = "forward"
        self.attention_score = 1.0
        self.attention_level = "high"
        
        # Eye tracking
        self.left_ear = 0.3
        self.right_ear = 0.3
        self.is_blinking = False
        self.blink_count = 0
        self.is_drowsy = False
        self.drowsiness_level = 0.0
        
        # Mouth tracking
        self.mar = 0.0
        self.is_yawning = False
        self.yawn_count = 0
        
        # Behavioral history
        self.pose_history = deque(maxlen=30)  # Last 30 frames
        self.attention_history = deque(maxlen=60)  # Last 60 frames (2 sec @ 30fps)
        
    def update(
        self,
        bbox: List[int],
        embedding: Optional[np.ndarray] = None,
        landmarks: Optional[List[List[int]]] = None,
        confidence: float = 0.0,
        image_shape: Optional[tuple] = None
    ):
        """
        Update track with behavioral analysis
        
        Args:
            bbox: Bounding box
            embedding: Face embedding
            landmarks: 5-point facial landmarks
            confidence: Detection confidence
            image_shape: (height, width) for pose estimation
        """
        # Update base track
        super().update(bbox, embedding, landmarks, confidence)
        
        # Analyze behavior if landmarks available
        if landmarks and len(landmarks) == 5:
            self._analyze_behavior(landmarks, image_shape)
    
    def _analyze_behavior(self, landmarks: List[List[int]], image_shape: Optional[tuple]):
        """
        Perform behavioral analysis
        
        Args:
            landmarks: 5-point facial landmarks
            image_shape: Image dimensions for pose estimation
        """
        # Head pose estimation
        if image_shape:
            self.yaw, self.pitch, self.roll = self.pose_estimator.estimate_pose(
                landmarks, image_shape
            )
            self.gaze_direction = get_gaze_direction(self.yaw, self.pitch)
            self.attention_score, self.attention_level = estimate_attention_level(
                self.yaw, self.pitch, self.roll
            )
            
            # Store pose history
            self.pose_history.append({
                'yaw': self.yaw,
                'pitch': self.pitch,
                'roll': self.roll,
                'attention': self.attention_score
            })
            
            self.attention_history.append(self.attention_score)
        
        # Eye tracking (blink/drowsiness detection)
        self.left_ear, self.right_ear = compute_ear_from_5_landmarks(landmarks)
        
        blink_result = self.blink_detector.update(self.left_ear, self.right_ear)
        self.is_blinking = blink_result['is_blinking']
        self.blink_count = blink_result['total_blinks']
        self.is_drowsy = blink_result['is_drowsy']
        self.drowsiness_level = blink_result['drowsiness_level']
        
        # Mouth tracking (yawn detection)
        # NOTE: Yawn detection with 5-point landmarks is VERY approximate
        # For production use, consider using 68-point facial landmarks
        self.mar = compute_mouth_aspect_ratio(landmarks)
        
        yawn_result = self.yawn_detector.update(self.mar)
        self.is_yawning = yawn_result['is_yawning']
        self.yawn_count = yawn_result['total_yawns']
    
    def get_average_attention(self, window: int = 30) -> float:
        """
        Get average attention over recent frames
        
        Args:
            window: Number of frames to average
        
        Returns:
            Average attention score (0.0-1.0)
        """
        if not self.attention_history:
            return self.attention_score
        
        recent = list(self.attention_history)[-window:]
        return sum(recent) / len(recent)
    
    def get_pose_stability(self, window: int = 30) -> float:
        """
        Calculate pose stability (variance in angles)
        
        Args:
            window: Number of frames to analyze
        
        Returns:
            Stability score (0.0-1.0, higher is more stable)
        """
        if len(self.pose_history) < 5:
            return 1.0
        
        recent = list(self.pose_history)[-window:]
        
        yaw_variance = np.var([p['yaw'] for p in recent])
        pitch_variance = np.var([p['pitch'] for p in recent])
        
        # Combined variance (normalized)
        total_variance = (yaw_variance + pitch_variance) / 2
        
        # Convert to stability (inverse, normalized)
        stability = 1.0 / (1.0 + total_variance / 100.0)
        
        return stability
    
    def is_attentive(self, threshold: float = 0.6) -> bool:
        """
        Check if person is currently attentive
        
        Args:
            threshold: Minimum attention score
        
        Returns:
            True if attentive
        """
        return self.attention_score >= threshold
    
    def get_engagement_level(self) -> str:
        """
        Get overall engagement level
        
        Returns:
            "engaged", "moderate", "disengaged"
        """
        avg_attention = self.get_average_attention(window=30)
        stability = self.get_pose_stability(window=30)
        
        # Combine attention and stability
        engagement = (avg_attention * 0.7 + stability * 0.3)
        
        # Factor in drowsiness
        if self.is_drowsy:
            engagement *= 0.5
        
        if engagement >= 0.7:
            return "engaged"
        elif engagement >= 0.4:
            return "moderate"
        else:
            return "disengaged"
    
    def to_dict(self) -> Dict:
        """Convert track with behavior to dictionary"""
        base_dict = super().to_dict()
        
        behavior_dict = {
            'yaw': self.yaw,
            'pitch': self.pitch,
            'roll': self.roll,
            'gaze_direction': self.gaze_direction,
            'attention_score': self.attention_score,
            'attention_level': self.attention_level,
            'average_attention': self.get_average_attention(),
            'is_blinking': self.is_blinking,
            'blink_count': self.blink_count,
            'is_drowsy': self.is_drowsy,
            'drowsiness_level': self.drowsiness_level,
            'is_yawning': self.is_yawning,
            'yawn_count': self.yawn_count,
            'engagement_level': self.get_engagement_level(),
            'pose_stability': self.get_pose_stability()
        }
        
        base_dict.update(behavior_dict)
        return base_dict
