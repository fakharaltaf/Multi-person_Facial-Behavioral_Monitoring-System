"""
Face alignment utilities
Aligns faces using 5-point landmarks for recognition
"""

import numpy as np
import cv2
from typing import List, Tuple

# Standard reference points for face alignment (112x112)
# Based on ArcFace training data alignment
REFERENCE_FACIAL_POINTS = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041]   # Right mouth corner
], dtype=np.float32)


def align_face(
    image: np.ndarray,
    landmarks: List[List[int]],
    output_size: Tuple[int, int] = (112, 112)
) -> np.ndarray:
    """
    Align face using 5-point landmarks
    
    Args:
        image: Input image
        landmarks: 5 facial landmarks [[x1,y1], [x2,y2], ...]
        output_size: Output image size (width, height)
    
    Returns:
        Aligned face image
    """
    if len(landmarks) != 5:
        raise ValueError(f"Expected 5 landmarks, got {len(landmarks)}")
    
    # Convert landmarks to numpy array
    src_points = np.array(landmarks, dtype=np.float32)
    
    # Scale reference points if output size differs from 112x112
    if output_size != (112, 112):
        scale_x = output_size[0] / 112.0
        scale_y = output_size[1] / 112.0
        dst_points = REFERENCE_FACIAL_POINTS.copy()
        dst_points[:, 0] *= scale_x
        dst_points[:, 1] *= scale_y
    else:
        dst_points = REFERENCE_FACIAL_POINTS
    
    # Estimate similarity transform
    transform_matrix = estimate_transform(src_points, dst_points)
    
    # Apply transformation
    aligned = cv2.warpAffine(
        image,
        transform_matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return aligned


def estimate_transform(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transformation matrix (rotation, scale, translation)
    
    Args:
        src_points: Source points (Nx2)
        dst_points: Destination points (Nx2)
    
    Returns:
        2x3 transformation matrix
    """
    assert src_points.shape == dst_points.shape
    assert src_points.shape[0] >= 3
    
    # Use cv2.estimateAffinePartial2D for similarity transform
    transform_matrix, _ = cv2.estimateAffinePartial2D(
        src_points,
        dst_points,
        method=cv2.LMEDS
    )
    
    if transform_matrix is None:
        # Fallback to basic affine transform
        transform_matrix = cv2.getAffineTransform(
            src_points[:3],
            dst_points[:3]
        )
    
    return transform_matrix


def align_faces_batch(
    image: np.ndarray,
    detections: List[dict],
    output_size: Tuple[int, int] = (112, 112)
) -> List[np.ndarray]:
    """
    Align multiple faces from detection results
    
    Args:
        image: Input image
        detections: List of detection dicts with 'landmarks' key
        output_size: Output image size
    
    Returns:
        List of aligned face images
    """
    aligned_faces = []
    
    for detection in detections:
        landmarks = detection.get('landmarks', [])
        if len(landmarks) == 5:
            try:
                aligned = align_face(image, landmarks, output_size)
                aligned_faces.append(aligned)
            except Exception as e:
                print(f"Warning: Failed to align face: {e}")
                continue
    
    return aligned_faces


def visualize_landmarks(image: np.ndarray, landmarks: List[List[int]]) -> np.ndarray:
    """
    Draw landmarks on image for visualization
    
    Args:
        image: Input image
        landmarks: 5 facial landmarks
    
    Returns:
        Image with landmarks drawn
    """
    vis_image = image.copy()
    
    # Define colors for each landmark
    colors = [
        (0, 255, 0),    # Left eye - Green
        (0, 255, 0),    # Right eye - Green
        (255, 0, 0),    # Nose - Blue
        (0, 0, 255),    # Left mouth - Red
        (0, 0, 255)     # Right mouth - Red
    ]
    
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(vis_image, (int(x), int(y)), 3, colors[i], -1)
        cv2.putText(
            vis_image,
            str(i+1),
            (int(x)+5, int(y)-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            colors[i],
            1
        )
    
    return vis_image
