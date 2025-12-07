"""
Detection module - Face and person detection
"""

from .retinaface import RetinaFace
from .utils import (
    draw_detections,
    filter_small_faces,
    calculate_iou,
    crop_face,
    get_face_center,
    get_face_size,
    sort_detections_by_size,
    sort_detections_by_confidence
)

__all__ = [
    'RetinaFace',
    'draw_detections',
    'filter_small_faces',
    'calculate_iou',
    'crop_face',
    'get_face_center',
    'get_face_size',
    'sort_detections_by_size',
    'sort_detections_by_confidence'
]
