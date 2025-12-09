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

# Try to import dlib pose estimator (optional)
try:
    from .dlib_head_pose import DlibHeadPoseEstimator
    DLIB_AVAILABLE = True
except:
    DLIB_AVAILABLE = False


class BehaviorTrack(Track):
    """
    Extended Track with behavioral analysis capabilities
    """
    
    def __init__(self, *args, use_dlib=False, dlib_estimator=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Behavioral analyzers
        self.pose_estimator = HeadPoseEstimator()
        self.use_dlib = use_dlib and DLIB_AVAILABLE
        self.dlib_estimator = dlib_estimator if self.use_dlib else None
        self.landmarks_68 = None  # Store 68-point landmarks if using dlib
        
        self.blink_detector = BlinkDetector()
        # Use lower threshold for yawn detection with 68-point landmarks (more accurate)
        # 5-point: 0.9 threshold (very conservative due to inaccuracy)
        # 68-point: 0.6 threshold (can be more sensitive with accurate landmarks)
        yawn_threshold = 0.6 if self.use_dlib else 0.9
        self.yawn_detector = YawnDetector(mar_threshold=yawn_threshold, consecutive_frames=6)
        
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
        image_shape: Optional[tuple] = None,
        image: Optional[np.ndarray] = None
    ):
        """
        Update track with behavioral analysis
        
        Args:
            bbox: Bounding box
            embedding: Face embedding
            landmarks: 5-point facial landmarks
            confidence: Detection confidence
            image_shape: (height, width) for pose estimation
            image: Full image for dlib landmark detection
        """
        # Update base track
        super().update(bbox, embedding, landmarks, confidence)
        
        # Try dlib first if enabled and image provided
        if self.use_dlib and self.dlib_estimator and image is not None:
            # Extract face region for dlib (expand bbox slightly for better detection)
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
                self.landmarks_68 = self.dlib_estimator.detect_landmarks(face_region, bbox=None)
                
                if self.landmarks_68 is not None:
                    # Adjust landmarks back to full image coordinates
                    self.landmarks_68[:, 0] += x1
                    self.landmarks_68[:, 1] += y1
                    
                    self._analyze_behavior_dlib(self.landmarks_68, image_shape)
                    return
        
        # Fall back to 5-point landmarks
        if landmarks and len(landmarks) == 5:
            self._analyze_behavior(landmarks, image_shape)
    
    def _analyze_behavior_dlib(self, landmarks_68: np.ndarray, image_shape: Optional[tuple]):
        """
        Perform behavioral analysis using 68-point dlib landmarks
        
        Args:
            landmarks_68: 68-point facial landmarks
            image_shape: Image dimensions for pose estimation
        """
        # Head pose estimation using dlib landmarks
        if image_shape:
            self.yaw, self.pitch, self.roll = self.dlib_estimator.estimate_pose(
                landmarks_68, image_shape
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
        
        # Eye tracking using dlib landmarks (much more accurate)
        # Left eye: landmarks 36-41, Right eye: landmarks 42-47
        left_eye = landmarks_68[36:42]
        right_eye = landmarks_68[42:48]
        
        self.left_ear = self._compute_ear(left_eye)
        self.right_ear = self._compute_ear(right_eye)
        
        blink_result = self.blink_detector.update(self.left_ear, self.right_ear)
        self.is_blinking = blink_result['is_blinking']
        self.blink_count = blink_result['total_blinks']
        self.is_drowsy = blink_result['is_drowsy']
        self.drowsiness_level = blink_result['drowsiness_level']
        
        # Mouth tracking using dlib landmarks (landmarks 48-67)
        mouth = landmarks_68[48:68]
        self.mar = self._compute_mar(mouth)
        
        yawn_result = self.yawn_detector.update(self.mar)
        self.is_yawning = yawn_result['is_yawning']
        self.yawn_count = yawn_result['total_yawns']
    
    def _compute_ear(self, eye_points: np.ndarray) -> float:
        """Compute Eye Aspect Ratio from 6 eye landmarks"""
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        # EAR
        return (v1 + v2) / (2.0 * h + 1e-6)
    
    def _compute_mar(self, mouth_points: np.ndarray) -> float:
        """Compute Mouth Aspect Ratio from mouth landmarks"""
        # Vertical distance (center of upper/lower lips)
        v1 = np.linalg.norm(mouth_points[13] - mouth_points[19])  # 61-67
        v2 = np.linalg.norm(mouth_points[14] - mouth_points[18])  # 62-66
        v3 = np.linalg.norm(mouth_points[15] - mouth_points[17])  # 63-65
        # Horizontal distance
        h = np.linalg.norm(mouth_points[0] - mouth_points[6])  # 48-54
        # MAR
        return (v1 + v2 + v3) / (3.0 * h + 1e-6)
    
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
