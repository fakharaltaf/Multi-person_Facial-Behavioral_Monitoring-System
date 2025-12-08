"""
Enhanced visualization with behavioral overlay
"""

import cv2
import numpy as np
from typing import List

from ..tracking.visualization import get_color_for_id
from ..behavior.behavior_track import BehaviorTrack


def draw_behavior_track(
    image: np.ndarray,
    track: BehaviorTrack,
    show_pose: bool = True,
    show_details: bool = True
) -> np.ndarray:
    """
    Draw track with behavioral information
    
    Args:
        image: Input image
        track: BehaviorTrack object
        show_pose: Show pose axes
        show_details: Show detailed behavioral info
    
    Returns:
        Image with behavioral visualization
    """
    if track.bbox is None:
        return image
    
    x1, y1, x2, y2 = [int(v) for v in track.bbox]
    color = get_color_for_id(track.track_id)
    
    # Draw bounding box with engagement color
    engagement = track.get_engagement_level()
    if engagement == "engaged":
        box_color = (0, 255, 0)  # Green
    elif engagement == "moderate":
        box_color = (0, 255, 255)  # Yellow
    else:
        box_color = (0, 0, 255)  # Red
    
    thickness = 3 if track.state == 'confirmed' else 1
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
    
    # Draw label
    if track.person_name and track.person_name != "Unknown":
        label = f"ID:{track.track_id} {track.person_name}"
    else:
        label = f"ID:{track.track_id}"
    
    # Background for text
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(
        image,
        (x1, y1 - label_size[1] - 10),
        (x1 + label_size[0], y1),
        box_color,
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
    
    # Draw behavioral info
    if show_details:
        info_y = y2 + 20
        
        # Attention
        att_text = f"Att: {track.attention_score:.2f} ({track.attention_level})"
        cv2.putText(image, att_text, (x1, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        info_y += 20
        
        # Gaze
        gaze_text = f"Gaze: {track.gaze_direction}"
        cv2.putText(image, gaze_text, (x1, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        info_y += 20
        
        # Pose angles
        pose_text = f"Y:{track.yaw:.0f} P:{track.pitch:.0f} R:{track.roll:.0f}"
        cv2.putText(image, pose_text, (x1, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
        info_y += 20
        
        # Alerts
        alerts = []
        if track.is_drowsy:
            alerts.append("DROWSY")
        if track.is_yawning:
            alerts.append("YAWN")
        if track.is_blinking:
            alerts.append("BLINK")
        
        if alerts:
            alert_text = " | ".join(alerts)
            cv2.putText(image, alert_text, (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw pose axes
    if show_pose and track.landmarks and len(track.landmarks) == 5:
        image = track.pose_estimator.draw_pose_axes(
            image,
            track.landmarks,
            track.yaw,
            track.pitch,
            track.roll,
            scale=80
        )
    
    return image


def draw_behavior_tracks(
    image: np.ndarray,
    tracks: List[BehaviorTrack],
    show_pose: bool = True,
    show_details: bool = True,
    show_stats: bool = True
) -> np.ndarray:
    """
    Draw all behavioral tracks
    
    Args:
        image: Input image
        tracks: List of BehaviorTrack objects
        show_pose: Show pose axes
        show_details: Show detailed info
        show_stats: Show statistics overlay
    
    Returns:
        Image with all tracks drawn
    """
    result = image.copy()
    
    # Draw each track
    for track in tracks:
        result = draw_behavior_track(result, track, show_pose, show_details)
    
    # Draw statistics overlay
    if show_stats and tracks:
        stats_x = 10
        stats_y = 30
        
        # Engagement statistics
        engagement_counts = {'engaged': 0, 'moderate': 0, 'disengaged': 0}
        drowsy_count = 0
        yawning_count = 0
        avg_attention = 0.0
        
        for track in tracks:
            engagement = track.get_engagement_level()
            engagement_counts[engagement] += 1
            avg_attention += track.attention_score
            if track.is_drowsy:
                drowsy_count += 1
            if track.is_yawning:
                yawning_count += 1
        
        avg_attention /= len(tracks) if tracks else 1
        
        # Draw stats background
        stats_height = 180
        stats_width = 280
        overlay = result.copy()
        cv2.rectangle(overlay, (stats_x - 5, stats_y - 25), 
                     (stats_x + stats_width, stats_y + stats_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
        
        # Stats text
        y_offset = stats_y
        cv2.putText(result, f"Active Tracks: {len(tracks)}", 
                   (stats_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(result, f"Engaged: {engagement_counts['engaged']}", 
                   (stats_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        
        cv2.putText(result, f"Moderate: {engagement_counts['moderate']}", 
                   (stats_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(result, f"Disengaged: {engagement_counts['disengaged']}", 
                   (stats_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25
        
        cv2.putText(result, f"Avg Attention: {avg_attention:.2f}", 
                   (stats_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        if drowsy_count > 0:
            cv2.putText(result, f"Drowsy: {drowsy_count}", 
                       (stats_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25
        
        if yawning_count > 0:
            cv2.putText(result, f"Yawning: {yawning_count}", 
                       (stats_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    return result


def create_behavior_visualization(
    image: np.ndarray,
    tracks: List[BehaviorTrack],
    fps: float = 0.0,
    show_pose: bool = True,
    show_details: bool = True
) -> np.ndarray:
    """
    Create comprehensive behavioral visualization
    
    Args:
        image: Input image
        tracks: List of behavior tracks
        fps: Frames per second
        show_pose: Show pose axes
        show_details: Show detailed info
    
    Returns:
        Visualization image
    """
    vis = draw_behavior_tracks(image, tracks, show_pose, show_details, show_stats=True)
    
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
