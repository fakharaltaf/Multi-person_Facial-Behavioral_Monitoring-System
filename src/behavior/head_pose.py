"""
Head pose estimation from facial landmarks
Estimates yaw, pitch, roll angles
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


# 3D model points of facial landmarks (normalized)
MODEL_POINTS_68 = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# Simplified 5-point model (for RetinaFace landmarks)
MODEL_POINTS_5 = np.array([
    (-73.53, 0.0, 0.0),          # Left eye
    (73.53, 0.0, 0.0),           # Right eye
    (0.0, -73.53, 0.0),          # Nose tip
    (-48.0, -146.0, 0.0),        # Left mouth corner
    (48.0, -146.0, 0.0)          # Right mouth corner
], dtype=np.float64)


class HeadPoseEstimator:
    """
    Estimates head pose (yaw, pitch, roll) from facial landmarks
    """
    
    def __init__(self, focal_length: Optional[float] = None):
        """
        Initialize head pose estimator
        
        Args:
            focal_length: Camera focal length (estimated from image if None)
        """
        self.focal_length = focal_length
        self.model_points = MODEL_POINTS_5
    
    def estimate_pose(
        self,
        landmarks: List[List[int]],
        image_shape: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose from 5-point landmarks
        
        Args:
            landmarks: 5 facial landmarks [[x, y], ...]
            image_shape: (height, width) of image
        
        Returns:
            (yaw, pitch, roll) in degrees
        """
        if len(landmarks) != 5:
            return 0.0, 0.0, 0.0
        
        height, width = image_shape
        
        # Convert landmarks to numpy array
        image_points = np.array(landmarks, dtype=np.float64)
        
        # Camera internals
        if self.focal_length is None:
            focal_length = width
        else:
            focal_length = self.focal_length
        
        center = (width / 2, height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
        
        # Convert rotation vector to angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        # Extract Euler angles
        yaw, pitch, roll = self._rotation_matrix_to_euler_angles(rotation_mat)
        
        return yaw, pitch, roll
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (yaw, pitch, roll)
        
        Args:
            R: 3x3 rotation matrix
        
        Returns:
            (yaw, pitch, roll) in degrees
        """
        # Calculate yaw
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(-R[1, 2], R[1, 1])
            roll = 0
        
        # Convert to degrees
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)
        
        return yaw, pitch, roll
    
    def draw_pose_axes(
        self,
        image: np.ndarray,
        landmarks: List[List[int]],
        yaw: float,
        pitch: float,
        roll: float,
        scale: float = 100.0
    ) -> np.ndarray:
        """
        Draw pose axes on image
        
        Args:
            image: Input image
            landmarks: 5 facial landmarks
            yaw, pitch, roll: Pose angles in degrees
            scale: Axis length scale
        
        Returns:
            Image with pose axes drawn
        """
        if len(landmarks) != 5:
            return image
        
        height, width = image.shape[:2]
        
        # Get nose tip position (landmark 2)
        nose = landmarks[2]
        
        # Camera parameters
        if self.focal_length is None:
            focal_length = width
        else:
            focal_length = self.focal_length
        
        center = (width / 2, height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        # Convert angles to rotation vector
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)
        
        # Rotation matrix from Euler angles
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        
        Ry = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])
        
        Rz = np.array([
            [np.cos(roll_rad), -np.sin(roll_rad), 0],
            [np.sin(roll_rad), np.cos(roll_rad), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx
        rotation_vec, _ = cv2.Rodrigues(R)
        
        # 3D axis points
        axis_points = np.array([
            [scale, 0, 0],    # X-axis (red)
            [0, scale, 0],    # Y-axis (green)
            [0, 0, scale]     # Z-axis (blue)
        ], dtype=np.float64)
        
        # Project to image
        image_points = np.array([landmarks[2]], dtype=np.float64)
        
        # Solve PnP to get translation
        success, _, translation_vec = cv2.solvePnP(
            self.model_points,
            np.array(landmarks, dtype=np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Project axis points
            axis_img_points, _ = cv2.projectPoints(
                axis_points,
                rotation_vec,
                translation_vec,
                camera_matrix,
                dist_coeffs
            )
            
            # Draw axes
            nose_point = tuple(map(int, nose))
            
            # X-axis (Red)
            x_point = tuple(map(int, axis_img_points[0].ravel()))
            cv2.line(image, nose_point, x_point, (0, 0, 255), 3)
            
            # Y-axis (Green)
            y_point = tuple(map(int, axis_img_points[1].ravel()))
            cv2.line(image, nose_point, y_point, (0, 255, 0), 3)
            
            # Z-axis (Blue)
            z_point = tuple(map(int, axis_img_points[2].ravel()))
            cv2.line(image, nose_point, z_point, (255, 0, 0), 3)
        
        return image


def get_gaze_direction(yaw: float, pitch: float) -> str:
    """
    Get simple gaze direction from pose angles
    
    Args:
        yaw: Head yaw angle in degrees
        pitch: Head pitch angle in degrees
    
    Returns:
        Direction string (e.g., "forward", "left", "up-right")
    """
    # Horizontal direction
    if yaw < -30:
        horizontal = "right"
    elif yaw > 30:
        horizontal = "left"
    else:
        horizontal = "forward"
    
    # Vertical direction
    if pitch < -20:
        vertical = "up"
    elif pitch > 20:
        vertical = "down"
    else:
        vertical = ""
    
    # Combine
    if vertical and horizontal != "forward":
        return f"{vertical}-{horizontal}"
    elif vertical:
        return vertical
    else:
        return horizontal


def estimate_attention_level(yaw: float, pitch: float, roll: float) -> Tuple[float, str]:
    """
    Estimate attention level based on pose angles
    
    Args:
        yaw, pitch, roll: Pose angles in degrees
    
    Returns:
        (attention_score, attention_level)
        attention_score: 0.0-1.0
        attention_level: "high", "medium", "low"
    """
    # Very lenient thresholds - within ±60° yaw and ±40° pitch is considered attentive
    
    # Calculate normalized deviations with generous thresholds
    yaw_dev = min(1.0, abs(yaw) / 60.0)  # Full penalty at 60° yaw (very lenient)
    pitch_dev = min(1.0, abs(pitch) / 40.0)  # Full penalty at 40° pitch
    roll_dev = min(1.0, abs(roll) / 50.0)  # Full penalty at 50° roll
    
    # Apply strong non-linear scaling (power of 1.5) to be very forgiving
    yaw_dev = np.power(yaw_dev, 1.5)
    pitch_dev = np.power(pitch_dev, 1.5)
    roll_dev = np.power(roll_dev, 1.5)
    
    # Combined deviation (lower weights for even more forgiveness)
    deviation = 0.5 * yaw_dev + 0.3 * pitch_dev + 0.2 * roll_dev
    deviation = min(1.0, deviation)
    
    # Attention score (inverse of deviation)
    attention = 1.0 - deviation
    
    # Add small boost to compensate for pose estimation noise/jitter
    # When face is mostly frontal, boost attention slightly
    if abs(yaw) < 15 and abs(pitch) < 15:
        attention = min(1.0, attention + 0.05)  # +5% boost for frontal poses
    
    # Ensure minimum score when looking roughly forward
    if abs(yaw) < 10 and abs(pitch) < 10:
        attention = max(attention, 0.95)  # Guarantee high score when very frontal
    
    # Categorize with lower thresholds
    if attention >= 0.5:
        level = "high"
    elif attention >= 0.25:
        level = "medium"
    else:
        level = "low"
    
    return attention, level
