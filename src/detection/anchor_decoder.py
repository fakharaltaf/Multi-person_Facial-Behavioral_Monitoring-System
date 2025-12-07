"""
Anchor-based detection decoder for RetinaFace
"""

import numpy as np
from typing import List, Tuple


def generate_anchors(base_size=16, ratios=[1.0], scales=[1, 2]):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchors = []
    for ratio in ratios:
        for scale in scales:
            w = base_size * scale * np.sqrt(ratio)
            h = base_size * scale / np.sqrt(ratio)
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return np.array(anchors)


def generate_anchor_centers(stride, size):
    """Generate anchor centers for a given stride and feature map size"""
    shifts_x = np.arange(0, size) * stride
    shifts_y = np.arange(0, size) * stride
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    return shifts


def anchors_plane(height, width, stride, base_anchors):
    """
    Generate all anchor boxes for a given feature map
    """
    A = base_anchors.shape[0]
    shifts = generate_anchor_centers(stride, width)
    K = shifts.shape[0]
    
    field_of_anchors = (
        base_anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    )
    field_of_anchors = field_of_anchors.reshape((K * A, 4))
    return field_of_anchors


def decode_bbox(bbox_pred, anchors, bbox_std=(0.1, 0.1, 0.2, 0.2)):
    """
    Decode bbox from predictions and anchors
    bbox_pred: [dx, dy, dw, dh]
    anchors: [x1, y1, x2, y2]
    bbox_std: Standard deviations for decoding (InsightFace uses 0.1, 0.1, 0.2, 0.2)
    """
    if bbox_pred.shape[0] == 0:
        return np.zeros((0, 4))
    
    # Convert anchors to center format
    widths = anchors[:, 2] - anchors[:, 0] + 1
    heights = anchors[:, 3] - anchors[:, 1] + 1
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights
    
    # Apply deltas with bbox_std
    dx = bbox_pred[:, 0] * bbox_std[0]
    dy = bbox_pred[:, 1] * bbox_std[1]
    dw = bbox_pred[:, 2] * bbox_std[2]
    dh = bbox_pred[:, 3] * bbox_std[3]
    
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights
    
    # Convert back to corner format
    pred_boxes = np.zeros_like(bbox_pred)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h  # y2
    
    return pred_boxes


def decode_landmarks(landmark_pred, anchors, landmark_std=0.1):
    """
    Decode landmarks from predictions and anchors
    landmark_std: Standard deviation for decoding (InsightFace uses 0.1)
    """
    if landmark_pred.shape[0] == 0:
        return np.zeros((0, 10))
    
    # Convert anchors to center format
    widths = anchors[:, 2] - anchors[:, 0] + 1
    heights = anchors[:, 3] - anchors[:, 1] + 1
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights
    
    pred_landmarks = np.zeros_like(landmark_pred)
    
    # Decode each landmark point with landmark_std
    for i in range(5):  # 5 landmarks
        pred_landmarks[:, i*2] = landmark_pred[:, i*2] * landmark_std * widths + ctr_x
        pred_landmarks[:, i*2+1] = landmark_pred[:, i*2+1] * landmark_std * heights + ctr_y
    
    return pred_landmarks
