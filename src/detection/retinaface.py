"""
RetinaFace ONNX Detector
Face detection with 5-point landmarks using ONNX Runtime
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path
from .anchor_decoder import generate_anchors, anchors_plane, decode_bbox, decode_landmarks

logger = logging.getLogger(__name__)


class RetinaFace:
    """
    RetinaFace face detector using ONNX Runtime
    
    Detects faces and returns bounding boxes with 5-point landmarks
    Optimized for AMD GPU with DirectML
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.25,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "dml"
    ):
        """
        Initialize RetinaFace detector
        
        Args:
            model_path: Path to ONNX model file
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for Non-Maximum Suppression (0.25 is very aggressive)
            input_size: Model input size (width, height)
            device: Device type ('dml' for DirectML, 'cpu' for CPU)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.device = device
        
        # Initialize ONNX session
        self.session = self._load_model()
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Generate anchors for multi-scale detection
        # RetinaFace uses 3 scales with strides [8, 16, 32]
        self.feature_stride_fpn = [8, 16, 32]
        self.anchor_cfg = {
            '32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': (1.0,), 'ALLOWED_BORDER': 9999},
            '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': (1.0,), 'ALLOWED_BORDER': 9999},
            '8': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': (1.0,), 'ALLOWED_BORDER': 9999},
        }
        self._init_anchors()
        
        logger.info(f"RetinaFace initialized: {model_path}")
        logger.info(f"Input: {self.input_name}, Outputs: {self.output_names}")
        logger.info(f"Device: {device}, Input size: {input_size}")
    
    def _init_anchors(self):
        """Initialize anchor boxes for all scales"""
        self.anchors_fpn = {}
        self.num_anchors = {}
        
        for stride in self.feature_stride_fpn:
            key = str(stride)
            cfg = self.anchor_cfg[key]
            base_anchors = generate_anchors(
                base_size=cfg['BASE_SIZE'],
                ratios=list(cfg['RATIOS']),
                scales=list(cfg['SCALES'])
            )
            
            # Calculate feature map size
            feat_h = self.input_size[1] // stride
            feat_w = self.input_size[0] // stride
            
            # Generate all anchors for this scale
            anchors = anchors_plane(feat_h, feat_w, stride, base_anchors)
            self.anchors_fpn[key] = anchors
            self.num_anchors[key] = base_anchors.shape[0]
            
            logger.debug(f"Generated {len(anchors)} anchors for stride {stride} (feature map: {feat_w}x{feat_h})")
    
    def _load_model(self) -> ort.InferenceSession:
        """Load ONNX model with appropriate execution provider"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Set up providers based on device
        if self.device == "dml":
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        elif self.device == "cpu":
            providers = ['CPUExecutionProvider']
        else:
            providers = [self.device]
        
        try:
            session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            actual_provider = session.get_providers()[0]
            logger.info(f"Model loaded with provider: {actual_provider}")
            return session
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Tuple of (preprocessed_image, scale, pad_info)
        """
        img_h, img_w = image.shape[:2]
        target_w, target_h = self.input_size
        
        # Calculate scaling to fit into target size while maintaining aspect ratio
        scale = min(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to RGB and normalize
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32)
        
        # Normalize (mean subtraction and scaling)
        padded = (padded - 127.5) / 128.0
        
        # Transpose to CHW format and add batch dimension
        padded = padded.transpose(2, 0, 1)
        padded = np.expand_dims(padded, axis=0)
        
        return padded, scale, (new_w, new_h)
    
    def _decode_multi_scale_outputs(
        self,
        scores_list: List[np.ndarray],
        bboxes_list: List[np.ndarray],
        landmarks_list: List[np.ndarray],
        scale: float,
        original_shape: Tuple[int, int]
    ) -> List[Dict]:
        """Decode multi-scale RetinaFace outputs using anchors"""
        detections = []
        orig_h, orig_w = original_shape
        
        # Process each scale (stride 32, 16, 8)
        for stride_idx, stride in enumerate(self.feature_stride_fpn):
            key = str(stride)
            anchors = self.anchors_fpn[key]
            num_anchors = self.num_anchors[key]
            
            scores = scores_list[stride_idx].flatten()
            bbox_deltas = bboxes_list[stride_idx].reshape((-1, 4))
            landmark_deltas = landmarks_list[stride_idx].reshape((-1, 10))
            
            # Filter by confidence threshold
            valid_indices = np.where(scores >= self.confidence_threshold)[0]
            
            if len(valid_indices) == 0:
                continue
            
            # Get valid predictions and anchors
            valid_scores = scores[valid_indices]
            valid_anchors = anchors[valid_indices]
            valid_bbox_deltas = bbox_deltas[valid_indices]
            valid_landmark_deltas = landmark_deltas[valid_indices]
            
            # Decode bounding boxes from deltas and anchors
            pred_bboxes = decode_bbox(valid_bbox_deltas, valid_anchors)
            
            # Decode landmarks from deltas and anchors
            pred_landmarks = decode_landmarks(valid_landmark_deltas, valid_anchors)
            
            # Scale back to original image coordinates
            for i in range(len(valid_scores)):
                bbox = pred_bboxes[i]
                landmark = pred_landmarks[i]
                score = float(valid_scores[i])
                
                # Scale bbox to original size
                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                
                # Clip to image boundaries
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Scale landmarks to original size
                scaled_lm = []
                for j in range(0, 10, 2):
                    lm_x = int(landmark[j] / scale)
                    lm_y = int(landmark[j+1] / scale)
                    lm_x = max(0, min(lm_x, orig_w - 1))
                    lm_y = max(0, min(lm_y, orig_h - 1))
                    scaled_lm.append([lm_x, lm_y])
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': score,
                    'landmarks': scaled_lm
                })
        
        return detections
    
    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        original_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        Postprocess model outputs to extract face detections
        
        Args:
            outputs: Raw model outputs
            scale: Scale factor used in preprocessing
            original_shape: Original image shape (height, width)
        
        Returns:
            List of detection dictionaries with bbox, landmarks, confidence
        """
        detections = []
        
        # InsightFace RetinaFace format (multi-scale):
        # 9 outputs in groups of 3:
        #   - 3 score outputs (one per scale)
        #   - 3 bbox outputs (one per scale)
        #   - 3 landmark outputs (one per scale)
        
        try:
            if len(outputs) != 9:
                logger.warning(f"Expected 9 outputs, got {len(outputs)}")
                return detections
            
            # Group outputs by type
            scores_list = [outputs[0], outputs[1], outputs[2]]  # Confidence scores
            bboxes_list = [outputs[3], outputs[4], outputs[5]]  # Bounding boxes
            landmarks_list = [outputs[6], outputs[7], outputs[8]]  # Landmarks
            
            # Decode detections from all scales
            detections = self._decode_multi_scale_outputs(
                scores_list,
                bboxes_list,
                landmarks_list,
                scale,
                original_shape
            )
        
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            logger.debug(f"Output shapes: {[out.shape for out in outputs]}")
            import traceback
            traceback.print_exc()
            return []
        
        # Apply Non-Maximum Suppression
        if len(detections) > 0:
            detections = self._nms(detections, self.nms_threshold)
        
        return detections
    
    def _nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        Uses multiple strategies to handle multi-scale detections
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for suppression
        
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []
        
        # Extract boxes and scores
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Calculate centers
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Check containment in both directions
            containment_in_current = inter / areas[order[1:]]  # How much of other boxes are inside current
            containment_of_current = inter / areas[i]  # How much of current box overlaps with others
            
            # Calculate center distance (normalized by box size)
            box_size_i = np.sqrt(areas[i])
            center_dist = np.sqrt((cx[i] - cx[order[1:]])**2 + (cy[i] - cy[order[1:]])**2)
            normalized_dist = center_dist / box_size_i
            
            # Remove boxes if ANY of these conditions are met:
            # 1. IoU exceeds threshold (standard NMS)
            # 2. Other box is >40% contained in current box (multi-scale issue)
            # 3. Current box is >30% overlapped with other
            # 4. Centers are very close (within 0.5 box sizes) regardless of IoU
            inds = np.where(
                (iou <= iou_threshold) & 
                (containment_in_current <= 0.4) & 
                (containment_of_current <= 0.3) &
                (normalized_dist > 0.5)
            )[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of detections with bbox, landmarks, confidence
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided")
            return []
        
        original_shape = image.shape[:2]
        
        # Preprocess
        input_data, scale, _ = self.preprocess(image)
        
        # Run inference
        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return []
        
        # Postprocess
        detections = self.postprocess(outputs, scale, original_shape)
        
        logger.debug(f"Detected {len(detections)} faces")
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect faces in batch of images
        
        Args:
            images: List of input images (BGR format)
        
        Returns:
            List of detection lists (one per image)
        """
        results = []
        for image in images:
            detections = self.detect(image)
            results.append(detections)
        return results
    
    def __call__(self, image: np.ndarray) -> List[Dict]:
        """Allow detector to be called directly"""
        return self.detect(image)
