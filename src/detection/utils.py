"""
Detection utility functions
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    draw_landmarks: bool = True,
    draw_confidence: bool = True,
    bbox_color: Tuple[int, int, int] = (0, 255, 0),
    landmark_color: Tuple[int, int, int] = (0, 0, 255),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw face detections on image
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        draw_landmarks: Whether to draw facial landmarks
        draw_confidence: Whether to draw confidence scores
        bbox_color: Bounding box color (B, G, R)
        landmark_color: Landmark color (B, G, R)
        text_color: Text color (B, G, R)
        thickness: Line thickness
    
    Returns:
        Image with drawn detections
    """
    vis_image = image.copy()
    
    for det in detections:
        # Draw bounding box
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), bbox_color, thickness)
        
        # Draw confidence score
        if draw_confidence:
            conf = det['confidence']
            text = f"{conf:.2f}"
            cv2.putText(
                vis_image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                thickness
            )
        
        # Draw landmarks
        if draw_landmarks and det.get('landmarks') is not None:
            landmarks = det['landmarks']
            for lm in landmarks:
                x, y = map(int, lm)
                cv2.circle(vis_image, (x, y), 2, landmark_color, -1)
    
    return vis_image


def filter_small_faces(
    detections: List[Dict],
    min_size: int = 30
) -> List[Dict]:
    """
    Filter out faces smaller than minimum size
    
    Args:
        detections: List of detection dictionaries
        min_size: Minimum face size in pixels
    
    Returns:
        Filtered detections
    """
    filtered = []
    for det in detections:
        bbox = det['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width >= min_size and height >= min_size:
            filtered.append(det)
    
    return filtered


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union of two bounding boxes
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def crop_face(
    image: np.ndarray,
    bbox: List[int],
    expand_ratio: float = 0.0
) -> np.ndarray:
    """
    Crop face region from image
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        expand_ratio: Ratio to expand bbox (e.g., 0.1 for 10% expansion)
    
    Returns:
        Cropped face image
    """
    x1, y1, x2, y2 = bbox
    
    if expand_ratio > 0:
        w = x2 - x1
        h = y2 - y1
        expand_w = int(w * expand_ratio / 2)
        expand_h = int(h * expand_ratio / 2)
        
        x1 = max(0, x1 - expand_w)
        y1 = max(0, y1 - expand_h)
        x2 = min(image.shape[1], x2 + expand_w)
        y2 = min(image.shape[0], y2 + expand_h)
    
    return image[y1:y2, x1:x2]


def get_face_center(bbox: List[int]) -> Tuple[int, int]:
    """
    Get center point of face bounding box
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Center coordinates (x, y)
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx, cy)


def get_face_size(bbox: List[int]) -> Tuple[int, int]:
    """
    Get width and height of face bounding box
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Size (width, height)
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return (width, height)


def sort_detections_by_size(
    detections: List[Dict],
    descending: bool = True
) -> List[Dict]:
    """
    Sort detections by face size
    
    Args:
        detections: List of detection dictionaries
        descending: Sort in descending order (largest first)
    
    Returns:
        Sorted detections
    """
    def get_area(det):
        bbox = det['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height
    
    return sorted(detections, key=get_area, reverse=descending)


def sort_detections_by_confidence(
    detections: List[Dict],
    descending: bool = True
) -> List[Dict]:
    """
    Sort detections by confidence score
    
    Args:
        detections: List of detection dictionaries
        descending: Sort in descending order (highest confidence first)
    
    Returns:
        Sorted detections
    """
    return sorted(detections, key=lambda x: x['confidence'], reverse=descending)
