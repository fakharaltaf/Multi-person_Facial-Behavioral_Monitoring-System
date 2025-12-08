"""
ArcFace face recognition model
Extracts face embeddings for identification
"""

import numpy as np
import cv2
import onnxruntime as ort
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class ArcFace:
    """
    ArcFace face recognition model using ONNX Runtime
    Extracts 512-dimensional face embeddings
    """
    
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (112, 112),
        device: str = "dml"
    ):
        """
        Initialize ArcFace model
        
        Args:
            model_path: Path to ONNX model file
            input_size: Model input size (width, height)
            device: Device type ('dml' for DirectML, 'cpu' for CPU)
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.device = device
        
        # Initialize ONNX session
        self.session = self._load_model()
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        logger.info(f"ArcFace initialized: {model_path}")
        logger.info(f"Input: {self.input_name}, Output: {self.output_name}")
        logger.info(f"Device: {device}, Input size: {input_size}")
    
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
    
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face for model input
        
        Args:
            face_image: Aligned face image (BGR)
        
        Returns:
            Preprocessed input tensor
        """
        # Resize if needed
        if face_image.shape[:2] != self.input_size[::-1]:
            face_image = cv2.resize(face_image, self.input_size)
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1] (ArcFace standard)
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 127.5
        
        # Transpose to CHW format
        face_chw = np.transpose(face_normalized, (2, 0, 1))
        
        # Add batch dimension
        face_batch = np.expand_dims(face_chw, axis=0)
        
        return face_batch.astype(np.float32)
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from aligned face image
        
        Args:
            face_image: Aligned face image (BGR, 112x112)
        
        Returns:
            512-dimensional embedding vector (L2 normalized)
        """
        if face_image is None or face_image.size == 0:
            logger.warning("Empty face image provided")
            return None
        
        # Preprocess
        input_data = self.preprocess(face_image)
        
        # Run inference
        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_data}
            )
            embedding = outputs[0].flatten()
            
            # L2 normalization
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error during embedding extraction: {e}")
            return None
    
    def extract_embeddings_batch(
        self,
        face_images: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Extract embeddings from multiple faces
        
        Args:
            face_images: List of aligned face images
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for face_image in face_images:
            embedding = self.extract_embedding(face_image)
            if embedding is not None:
                embeddings.append(embedding)
        
        return embeddings
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Cosine similarity (embeddings are already L2 normalized)
        similarity = np.dot(embedding1, embedding2)
        
        # Convert from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def find_best_match(
        self,
        query_embedding: np.ndarray,
        gallery_embeddings: List[np.ndarray],
        threshold: float = 0.6
    ) -> Tuple[Optional[int], float]:
        """
        Find best matching face in gallery
        
        Args:
            query_embedding: Query face embedding
            gallery_embeddings: List of gallery face embeddings
            threshold: Minimum similarity threshold
        
        Returns:
            (best_match_index, similarity) or (None, 0.0) if no match
        """
        if not gallery_embeddings or query_embedding is None:
            return None, 0.0
        
        best_idx = None
        best_sim = 0.0
        
        for idx, gallery_emb in enumerate(gallery_embeddings):
            sim = self.compute_similarity(query_embedding, gallery_emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        
        if best_sim >= threshold:
            return best_idx, best_sim
        else:
            return None, best_sim
