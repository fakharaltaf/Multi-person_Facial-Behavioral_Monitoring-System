"""
Improved head pose estimation using dlib's 68-point facial landmarks
Much more accurate than 5-point RetinaFace landmarks
"""

import numpy as np
import cv2
import dlib
from typing import List, Tuple, Optional

# 3D model points for 68-point landmarks (key points for pose estimation)
# Using standard facial feature points
MODEL_POINTS_68 = np.array([
    (0.0, 0.0, 0.0),             # 30 - Nose tip
    (0.0, -330.0, -65.0),        # 8 - Chin
    (-225.0, 170.0, -135.0),     # 36 - Left eye left corner
    (225.0, 170.0, -135.0),      # 45 - Right eye right corner
    (-150.0, -150.0, -125.0),    # 48 - Left mouth corner
    (150.0, -150.0, -125.0)      # 54 - Right mouth corner
], dtype=np.float64)


class DlibHeadPoseEstimator:
    """
    Head pose estimator using dlib's 68-point facial landmarks
    Provides much more accurate yaw/pitch/roll angles than 5-point approach
    """
    
    def __init__(self, predictor_path: str = "models/shape_predictor_68_face_landmarks.dat"):
        """
        Initialize dlib head pose estimator
        
        Args:
            predictor_path: Path to dlib's shape predictor model
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.model_points = MODEL_POINTS_68
        
    def detect_landmarks(self, image: np.ndarray, bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """
        Detect 68 facial landmarks using dlib
        
        Args:
            image: Input image (BGR or grayscale)
            bbox: Optional face bounding box [x1, y1, x2, y2]. If None, uses dlib's detector
        
        Returns:
            68x2 array of landmark coordinates, or None if failed
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # If bbox provided, use it; otherwise detect face with dlib
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
            else:
                # Use dlib's own face detector
                dets = self.detector(gray, 1)
                if len(dets) == 0:
                    return None
                rect = dets[0]
            
            # Detect landmarks
            shape = self.predictor(gray, rect)
            
            # Convert to numpy array
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            
            return landmarks
        except Exception as e:
            print(f"Landmark detection failed: {e}")
            return None
    
    def estimate_pose(
        self,
        landmarks_68: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose from 68-point landmarks
        
        Args:
            landmarks_68: 68x2 array of facial landmarks
            image_shape: (height, width) of image
        
        Returns:
            (yaw, pitch, roll) in degrees
        """
        if landmarks_68 is None or len(landmarks_68) != 68:
            return 0.0, 0.0, 0.0
        
        height, width = image_shape
        
        # Select key points from 68 landmarks for pose estimation
        # Using: nose tip (30), chin (8), left eye corner (36), right eye corner (45),
        #        left mouth corner (48), right mouth corner (54)
        image_points = np.array([
            landmarks_68[30],  # Nose tip
            landmarks_68[8],   # Chin
            landmarks_68[36],  # Left eye left corner
            landmarks_68[45],  # Right eye right corner
            landmarks_68[48],  # Left mouth corner
            landmarks_68[54]   # Right mouth corner
        ], dtype=np.float64)
        
        # Camera internals (simple approximation)
        focal_length = width
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
        Uses the same convention as the 5-point estimator
        
        Args:
            R: 3x3 rotation matrix
        
        Returns:
            (yaw, pitch, roll) in degrees
        """
        # Use atan2 for more stable angle extraction
        # This matches the convention used in most computer vision applications
        
        # Extract pitch (rotation around X axis)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            # Non-gimbal lock case
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            # Gimbal lock case
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(-R[1, 2], R[1, 1])
            roll = 0
        
        # Convert to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)
        
        return yaw, pitch, roll
    
    def draw_pose_axes(
        self,
        image: np.ndarray,
        landmarks_68: np.ndarray,
        yaw: float,
        pitch: float,
        roll: float,
        scale: float = 100.0
    ) -> np.ndarray:
        """
        Draw pose axes on image
        
        Args:
            image: Input image
            landmarks_68: 68 facial landmarks
            yaw, pitch, roll: Pose angles in degrees
            scale: Axis length scale
        
        Returns:
            Image with pose axes drawn
        """
        if landmarks_68 is None or len(landmarks_68) != 68:
            return image
        
        height, width = image.shape[:2]
        
        # Use nose tip as origin
        nose = landmarks_68[30]
        
        # Camera parameters
        focal_length = width
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
        
        # Rotation matrices
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
        
        # Get key points for solvePnP
        image_points = np.array([
            landmarks_68[30],  # Nose tip
            landmarks_68[8],   # Chin
            landmarks_68[36],  # Left eye left corner
            landmarks_68[45],  # Right eye right corner
            landmarks_68[48],  # Left mouth corner
            landmarks_68[54]   # Right mouth corner
        ], dtype=np.float64)
        
        # Solve PnP to get translation
        success, _, translation_vec = cv2.solvePnP(
            self.model_points,
            image_points,
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
