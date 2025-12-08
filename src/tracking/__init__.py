"""
Tracking module - Multi-object tracking for faces
"""

from .track import Track, compute_iou, compute_embedding_distance, compute_center_distance
from .tracker import MultiObjectTracker

__all__ = [
    'Track',
    'MultiObjectTracker',
    'compute_iou',
    'compute_embedding_distance',
    'compute_center_distance'
]
