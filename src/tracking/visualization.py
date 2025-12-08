"""
Visualization utilities for tracking
"""

import cv2
import numpy as np
from typing import List, Dict
from ..tracking.track import Track


# Generate distinct colors for different track IDs
def get_color_for_id(track_id: int) -> tuple:
    """Generate a consistent color for a track ID"""
    np.random.seed(track_id)
    color = tuple(np.random.randint(0, 255, 3).tolist())
    np.random.seed()  # Reset seed
    return color


def draw_track(image: np.ndarray, track: Track, show_trajectory: bool = True) -> np.ndarray:
    """
    Draw a single track on image
    
    Args:
        image: Input image
        track: Track object
        show_trajectory: Whether to draw trajectory history
    
    Returns:
        Image with track drawn
    """
    if track.bbox is None:
        return image
    
    x1, y1, x2, y2 = [int(v) for v in track.bbox]
    color = get_color_for_id(track.track_id)
    
    # Draw bounding box
    thickness = 3 if track.state == 'confirmed' else 1
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label
    if track.person_name and track.person_name != "Unknown":
        label = f"ID:{track.track_id} {track.person_name}"
        conf_text = f"{track.identity_confidence:.2f}"
    else:
        label = f"ID:{track.track_id}"
        conf_text = f"{track.confidence:.2f}"
    
    # Background for text
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(
        image,
        (x1, y1 - label_size[1] - 10),
        (x1 + label_size[0], y1),
        color,
        -1
    )
    
    # Text
    cv2.putText(
        image,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    # Draw confidence
    cv2.putText(
        image,
        conf_text,
        (x1, y2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )
    
    # Draw landmarks if available
    if track.landmarks:
        for lm_x, lm_y in track.landmarks:
            cv2.circle(image, (int(lm_x), int(lm_y)), 2, color, -1)
    
    # Draw trajectory
    if show_trajectory and len(track.position_history) > 1:
        points = [(int(x), int(y)) for x, y in track.position_history]
        for i in range(len(points) - 1):
            alpha = i / len(points)  # Fade older points
            thickness = max(1, int(3 * alpha))
            cv2.line(image, points[i], points[i + 1], color, thickness)
    
    # Draw velocity arrow
    if track.velocity is not None and np.linalg.norm(track.velocity) > 1:
        center = track.get_center()
        if center:
            cx, cy = int(center[0]), int(center[1])
            vx, vy = track.velocity
            end_x = int(cx + vx * 5)
            end_y = int(cy + vy * 5)
            cv2.arrowedLine(image, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)
    
    return image


def draw_tracks(
    image: np.ndarray,
    tracks: List[Track],
    show_trajectory: bool = True,
    show_stats: bool = True
) -> np.ndarray:
    """
    Draw all tracks on image
    
    Args:
        image: Input image
        tracks: List of Track objects
        show_trajectory: Whether to draw trajectory history
        show_stats: Whether to show statistics overlay
    
    Returns:
        Image with all tracks drawn
    """
    result = image.copy()
    
    # Draw each track
    for track in tracks:
        result = draw_track(result, track, show_trajectory)
    
    # Draw statistics overlay
    if show_stats:
        stats_y = 30
        
        # Total tracks
        cv2.putText(
            result,
            f"Active Tracks: {len(tracks)}",
            (10, stats_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Identified persons
        identified = len([t for t in tracks if t.person_id is not None])
        cv2.putText(
            result,
            f"Identified: {identified}",
            (10, stats_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    
    return result


def draw_grid_overlay(image: np.ndarray, grid_size: int = 50) -> np.ndarray:
    """Draw a grid overlay for spatial reference"""
    h, w = image.shape[:2]
    
    # Vertical lines
    for x in range(0, w, grid_size):
        cv2.line(image, (x, 0), (x, h), (100, 100, 100), 1)
    
    # Horizontal lines
    for y in range(0, h, grid_size):
        cv2.line(image, (0, y), (w, y), (100, 100, 100), 1)
    
    return image


def create_tracking_visualization(
    image: np.ndarray,
    tracks: List[Track],
    fps: float = 0.0,
    show_trajectory: bool = True,
    show_grid: bool = False
) -> np.ndarray:
    """
    Create comprehensive tracking visualization
    
    Args:
        image: Input image
        tracks: List of tracks
        fps: Frames per second
        show_trajectory: Show trajectory trails
        show_grid: Show spatial grid
    
    Returns:
        Visualization image
    """
    vis = image.copy()
    
    # Grid overlay
    if show_grid:
        vis = draw_grid_overlay(vis, grid_size=50)
    
    # Draw tracks
    vis = draw_tracks(vis, tracks, show_trajectory=show_trajectory, show_stats=True)
    
    # FPS counter
    if fps > 0:
        cv2.putText(
            vis,
            f"FPS: {fps:.1f}",
            (image.shape[1] - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
    
    return vis
