"""
Enhanced visualization with behavioral overlay
"""

import cv2
import numpy as np
from typing import List

from ..tracking.visualization import get_color_for_id
from ..behavior.behavior_track import BehaviorTrack


def draw_68_landmarks(
    image: np.ndarray,
    landmarks_68: np.ndarray,
    color: tuple = (0, 255, 0),
    radius: int = 2
) -> np.ndarray:
    """
    Draw 68 facial landmarks on image
    
    Args:
        image: Input image
        landmarks_68: 68x2 array of landmark coordinates
        color: Color for landmarks (B, G, R)
        radius: Radius of landmark points
    
    Returns:
        Image with landmarks drawn
    """
    if landmarks_68 is None or len(landmarks_68) != 68:
        return image
    
    result = image.copy()
    
    # Draw all landmarks
    for i, (x, y) in enumerate(landmarks_68):
        cv2.circle(result, (int(x), int(y)), radius, color, -1)
    
    # Highlight key points used for pose estimation in different color
    key_points = [30, 8, 36, 45, 48, 54]  # nose, chin, eye corners, mouth corners
    for idx in key_points:
        x, y = landmarks_68[idx]
        cv2.circle(result, (int(x), int(y)), radius + 2, (0, 0, 255), -1)
    
    # Draw facial feature contours
    # Jaw line (0-16)
    for i in range(16):
        pt1 = tuple(landmarks_68[i].astype(int))
        pt2 = tuple(landmarks_68[i + 1].astype(int))
        cv2.line(result, pt1, pt2, color, 1)
    
    # Eyebrows
    for i in range(17, 21):  # Left eyebrow
        pt1 = tuple(landmarks_68[i].astype(int))
        pt2 = tuple(landmarks_68[i + 1].astype(int))
        cv2.line(result, pt1, pt2, color, 1)
    
    for i in range(22, 26):  # Right eyebrow
        pt1 = tuple(landmarks_68[i].astype(int))
        pt2 = tuple(landmarks_68[i + 1].astype(int))
        cv2.line(result, pt1, pt2, color, 1)
    
    # Nose
    for i in range(27, 30):  # Nose bridge
        pt1 = tuple(landmarks_68[i].astype(int))
        pt2 = tuple(landmarks_68[i + 1].astype(int))
        cv2.line(result, pt1, pt2, color, 1)
    
    for i in range(31, 35):  # Nose base
        pt1 = tuple(landmarks_68[i].astype(int))
        pt2 = tuple(landmarks_68[i + 1].astype(int))
        cv2.line(result, pt1, pt2, color, 1)
    
    # Eyes
    for i in range(36, 41):  # Left eye
        pt1 = tuple(landmarks_68[i].astype(int))
        pt2 = tuple(landmarks_68[i + 1].astype(int))
        cv2.line(result, pt1, pt2, color, 1)
    pt1 = tuple(landmarks_68[41].astype(int))
    pt2 = tuple(landmarks_68[36].astype(int))
    cv2.line(result, pt1, pt2, color, 1)
    
    for i in range(42, 47):  # Right eye
        pt1 = tuple(landmarks_68[i].astype(int))
        pt2 = tuple(landmarks_68[i + 1].astype(int))
        cv2.line(result, pt1, pt2, color, 1)
    pt1 = tuple(landmarks_68[47].astype(int))
    pt2 = tuple(landmarks_68[42].astype(int))
    cv2.line(result, pt1, pt2, color, 1)
    
    # Mouth outer
    for i in range(48, 59):
        pt1 = tuple(landmarks_68[i].astype(int))
        pt2 = tuple(landmarks_68[i + 1].astype(int))
        cv2.line(result, pt1, pt2, color, 1)
    pt1 = tuple(landmarks_68[59].astype(int))
    pt2 = tuple(landmarks_68[48].astype(int))
    cv2.line(result, pt1, pt2, color, 1)
    
    # Mouth inner
    for i in range(60, 67):
        pt1 = tuple(landmarks_68[i].astype(int))
        pt2 = tuple(landmarks_68[i + 1].astype(int))
        cv2.line(result, pt1, pt2, color, 1)
    pt1 = tuple(landmarks_68[67].astype(int))
    pt2 = tuple(landmarks_68[60].astype(int))
    cv2.line(result, pt1, pt2, color, 1)
    
    return result


def draw_behavior_track(
    image: np.ndarray,
    track: BehaviorTrack,
    show_pose: bool = True,
    show_details: bool = True,
    show_landmarks: bool = False
) -> np.ndarray:
    """
    Draw track with behavioral information
    
    Args:
        image: Input image
        track: BehaviorTrack object
        show_pose: Show pose axes
        show_details: Show detailed behavioral info
        show_landmarks: Show 68-point landmarks if available
    
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
    
    # Draw 68-point landmarks if available and enabled
    if show_landmarks and hasattr(track, 'landmarks_68') and track.landmarks_68 is not None:
        image = draw_68_landmarks(image, track.landmarks_68, color=(0, 255, 0), radius=1)
    
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
    show_stats: bool = True,
    show_landmarks: bool = False
) -> np.ndarray:
    """
    Draw all behavioral tracks
    
    Args:
        image: Input image
        tracks: List of BehaviorTrack objects
        show_pose: Show pose axes
        show_details: Show detailed info
        show_stats: Show statistics overlay
        show_landmarks: Show 68-point landmarks if available
    
    Returns:
        Image with all tracks drawn
    """
    result = image.copy()
    
    # Draw each track
    for track in tracks:
        result = draw_behavior_track(result, track, show_pose, show_details, show_landmarks)
    
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
    show_details: bool = True,
    show_landmarks: bool = False
) -> np.ndarray:
    """
    Create comprehensive behavioral visualization with metrics panel below video
    
    Args:
        image: Input image
        tracks: List of behavior tracks
        fps: Frames per second
        show_pose: Show pose axes
        show_details: Show detailed info
    
    Returns:
        Visualization image with metrics panel
    """
    # Draw tracks on original image (without overlaid stats)
    vis = image.copy()
    
    # Draw each track
    for track in tracks:
        vis = draw_behavior_track(vis, track, show_pose, show_details=False, show_landmarks=show_landmarks)
    
    # FPS counter (top right)
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
    
    # Create metrics panel below the video
    panel_height = 200
    panel = np.zeros((panel_height, image.shape[1], 3), dtype=np.uint8)
    panel.fill(30)  # Dark gray background
    
    if tracks:
        # Left column: Overall statistics
        col1_x = 20
        y = 30
        
        cv2.putText(panel, "OVERALL STATISTICS", (col1_x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        
        # Engagement stats
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
        
        avg_attention /= len(tracks)
        
        cv2.putText(panel, f"Active: {len(tracks)}", (col1_x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(panel, f"Engaged: {engagement_counts['engaged']}", (col1_x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 25
        cv2.putText(panel, f"Moderate: {engagement_counts['moderate']}", (col1_x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 25
        cv2.putText(panel, f"Disengaged: {engagement_counts['disengaged']}", (col1_x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y += 25
        cv2.putText(panel, f"Avg Attention: {avg_attention:.3f}", (col1_x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Right columns: Individual track details
        col_width = 280
        start_col = 300
        
        for idx, track in enumerate(tracks[:3]):  # Show up to 3 tracks
            col_x = start_col + (idx * col_width)
            y = 30
            
            # Header
            name = track.person_name if track.person_name != "Unknown" else f"Track {track.track_id}"
            cv2.putText(panel, name.upper(), (col_x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y += 30
            
            # Head pose angles
            cv2.putText(panel, f"Yaw:   {track.yaw:6.1f} deg", (col_x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 23
            cv2.putText(panel, f"Pitch: {track.pitch:6.1f} deg", (col_x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 23
            cv2.putText(panel, f"Roll:  {track.roll:6.1f} deg", (col_x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 23
            
            # Gaze and attention
            cv2.putText(panel, f"Gaze: {track.gaze_direction}", (col_x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 23
            
            att_color = (0, 255, 0) if track.attention_score >= 0.7 else (0, 255, 255) if track.attention_score >= 0.5 else (0, 0, 255)
            cv2.putText(panel, f"Attention: {track.attention_score:.3f}", (col_x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, att_color, 1)
            y += 23
            
            # Alerts
            if track.is_drowsy:
                cv2.putText(panel, "DROWSY", (col_x, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            if track.is_yawning:
                cv2.putText(panel, "YAWNING", (col_x + 80, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)
    else:
        # No tracks detected
        cv2.putText(panel, "No faces detected", (image.shape[1]//2 - 100, panel_height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
    
    # Stack video and panel vertically
    result = np.vstack([vis, panel])
    
    return result
