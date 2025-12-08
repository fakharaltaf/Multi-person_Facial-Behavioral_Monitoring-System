"""
Eye aspect ratio (EAR) for blink and drowsiness detection
"""

import numpy as np
from typing import List, Tuple
from collections import deque


def compute_eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio (EAR) from eye landmarks
    
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Args:
        eye_landmarks: 6 eye landmarks in order [outer, top1, top2, inner, bottom2, bottom1]
    
    Returns:
        EAR value (typically 0.2-0.4 for open eyes, <0.2 for closed)
    """
    if len(eye_landmarks) != 6:
        return 0.3  # Default value
    
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # EAR
    if C == 0:
        return 0.3
    
    ear = (A + B) / (2.0 * C)
    
    return ear


def compute_ear_from_5_landmarks(landmarks: List[List[int]]) -> Tuple[float, float]:
    """
    Approximate EAR from 5-point RetinaFace landmarks
    
    Args:
        landmarks: 5 facial landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
    
    Returns:
        (left_ear, right_ear) - approximated EAR values
    """
    if len(landmarks) != 5:
        return 0.3, 0.3
    
    left_eye = np.array(landmarks[0])
    right_eye = np.array(landmarks[1])
    nose = np.array(landmarks[2])
    
    # Estimate eye height based on eye-nose distance
    left_eye_to_nose = np.linalg.norm(left_eye - nose)
    right_eye_to_nose = np.linalg.norm(right_eye - nose)
    
    # Estimate eye width (inter-eye distance / 4)
    inter_eye = np.linalg.norm(right_eye - left_eye)
    eye_width = inter_eye / 4
    
    # Approximate EAR (assumes standard proportions)
    # Normal open eye has EAR ~ 0.3
    # This is a rough approximation
    left_ear = min(0.4, left_eye_to_nose / (eye_width + 1e-6) * 0.15)
    right_ear = min(0.4, right_eye_to_nose / (eye_width + 1e-6) * 0.15)
    
    return left_ear, right_ear


class BlinkDetector:
    """
    Detects blinks and estimates drowsiness from eye aspect ratio
    """
    
    def __init__(
        self,
        ear_threshold: float = 0.21,
        consecutive_frames: int = 2,
        drowsiness_threshold: float = 0.25,
        drowsiness_frames: int = 20
    ):
        """
        Initialize blink detector
        
        Args:
            ear_threshold: EAR below this is considered closed eye
            consecutive_frames: Minimum consecutive frames for blink
            drowsiness_threshold: Average EAR below this indicates drowsiness
            drowsiness_frames: Number of frames to average for drowsiness
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.drowsiness_threshold = drowsiness_threshold
        
        self.ear_history = deque(maxlen=drowsiness_frames)
        self.blink_counter = 0
        self.total_blinks = 0
        self.is_blinking = False
    
    def update(self, left_ear: float, right_ear: float) -> dict:
        """
        Update with new EAR values
        
        Args:
            left_ear: Left eye aspect ratio
            right_ear: Right eye aspect ratio
        
        Returns:
            Status dict with blink and drowsiness info
        """
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Add to history
        self.ear_history.append(avg_ear)
        
        # Detect blink
        blink_detected = False
        if avg_ear < self.ear_threshold:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.consecutive_frames:
                self.total_blinks += 1
                blink_detected = True
            self.blink_counter = 0
        
        self.is_blinking = self.blink_counter >= self.consecutive_frames
        
        # Check drowsiness
        is_drowsy = False
        drowsiness_level = 0.0
        
        if len(self.ear_history) >= 10:
            avg_ear_history = sum(self.ear_history) / len(self.ear_history)
            
            if avg_ear_history < self.drowsiness_threshold:
                is_drowsy = True
                # Drowsiness level: 0.0 (awake) to 1.0 (very drowsy)
                drowsiness_level = 1.0 - (avg_ear_history / self.drowsiness_threshold)
                drowsiness_level = min(1.0, max(0.0, drowsiness_level))
        
        return {
            'ear': avg_ear,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'is_blinking': self.is_blinking,
            'blink_detected': blink_detected,
            'total_blinks': self.total_blinks,
            'is_drowsy': is_drowsy,
            'drowsiness_level': drowsiness_level
        }
    
    def reset(self):
        """Reset detector state"""
        self.ear_history.clear()
        self.blink_counter = 0
        self.total_blinks = 0
        self.is_blinking = False


def compute_mouth_aspect_ratio(landmarks: List[List[int]]) -> float:
    """
    Compute Mouth Aspect Ratio (MAR) for yawn detection
    
    Note: With only 5 landmarks (RetinaFace), we cannot accurately measure
    mouth opening. This function provides a rough estimate based on facial
    geometry, but it's NOT reliable for yawn detection.
    
    Args:
        landmarks: 5 facial landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
    
    Returns:
        Estimated MAR value (very approximate)
    """
    if len(landmarks) != 5:
        return 0.0
    
    left_eye = np.array(landmarks[0])
    right_eye = np.array(landmarks[1])
    nose = np.array(landmarks[2])
    left_mouth = np.array(landmarks[3])
    right_mouth = np.array(landmarks[4])
    
    # Face height (eye to mouth distance)
    eye_center = (left_eye + right_eye) / 2
    mouth_center = (left_mouth + right_mouth) / 2
    face_height = np.linalg.norm(mouth_center - eye_center)
    
    # Mouth width
    mouth_width = np.linalg.norm(right_mouth - left_mouth)
    
    # Eye-to-eye distance (face width reference)
    eye_width = np.linalg.norm(right_eye - left_eye)
    
    # Estimate MAR based on mouth width relative to face proportions
    # Normal: mouth_width ~= 0.6 * eye_width
    # Yawning: mouth_width ~= 0.9-1.2 * eye_width
    if eye_width == 0:
        return 0.0
    
    mouth_ratio = mouth_width / eye_width
    
    # Scale to approximate MAR range (normal ~0.3-0.5, yawn >0.8)
    # This is still very approximate!
    mar = mouth_ratio * 0.7
    
    return mar


class YawnDetector:
    """
    Detects yawns from mouth aspect ratio
    
    WARNING: With only 5-point landmarks, yawn detection is VERY unreliable.
    The thresholds are set very high to avoid false positives.
    For accurate yawn detection, use 68-point facial landmarks.
    """
    
    def __init__(
        self,
        mar_threshold: float = 0.9,  # Very high threshold to reduce false positives
        consecutive_frames: int = 8  # Require more frames to confirm yawn
    ):
        """
        Initialize yawn detector
        
        Args:
            mar_threshold: MAR above this indicates yawn (set high for 5-point landmarks)
            consecutive_frames: Minimum consecutive frames for yawn
        """
        self.mar_threshold = mar_threshold
        self.consecutive_frames = consecutive_frames
        
        self.yawn_counter = 0
        self.total_yawns = 0
        self.is_yawning = False
    
    def update(self, mar: float) -> dict:
        """
        Update with new MAR value
        
        Args:
            mar: Mouth aspect ratio
        
        Returns:
            Status dict with yawn info
        """
        yawn_detected = False
        
        if mar > self.mar_threshold:
            self.yawn_counter += 1
        else:
            if self.yawn_counter >= self.consecutive_frames:
                self.total_yawns += 1
                yawn_detected = True
            self.yawn_counter = 0
        
        self.is_yawning = self.yawn_counter >= self.consecutive_frames
        
        return {
            'mar': mar,
            'is_yawning': self.is_yawning,
            'yawn_detected': yawn_detected,
            'total_yawns': self.total_yawns
        }
    
    def reset(self):
        """Reset detector state"""
        self.yawn_counter = 0
        self.total_yawns = 0
        self.is_yawning = False
