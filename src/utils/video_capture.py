"""
Video capture utility with support for webcam, video files, and IP cameras
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VideoCapture:
    """Unified video capture interface"""
    
    def __init__(
        self,
        source: str = "webcam",
        camera_id: int = 0,
        video_path: Optional[str] = None,
        ip_camera_url: Optional[str] = None,
        resolution: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize video capture
        
        Args:
            source: Source type ('webcam', 'video', 'ip_camera')
            camera_id: Camera device ID
            video_path: Path to video file
            ip_camera_url: URL for IP camera (RTSP/HTTP)
            resolution: Target resolution (width, height)
        """
        self.source = source
        self.resolution = resolution
        self.cap = None
        self.fps = 30  # Default FPS
        self.frame_count = 0
        self.total_frames = -1
        
        # Initialize capture based on source
        if source == "webcam":
            self.cap = cv2.VideoCapture(camera_id)
            logger.info(f"Initialized webcam (device {camera_id})")
        
        elif source == "video":
            if video_path is None:
                raise ValueError("video_path must be provided for video source")
            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Initialized video file: {video_path} ({self.total_frames} frames)")
        
        elif source == "ip_camera":
            if ip_camera_url is None:
                raise ValueError("ip_camera_url must be provided for IP camera source")
            self.cap = cv2.VideoCapture(ip_camera_url)
            logger.info(f"Initialized IP camera: {ip_camera_url}")
        
        else:
            raise ValueError(f"Unknown source type: {source}")
        
        # Check if capture opened successfully
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        # Get actual FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or self.fps > 120:  # Sanity check
            self.fps = 30
            logger.warning(f"Invalid FPS detected, using default: {self.fps}")
        
        # Set resolution if specified
        if resolution:
            self.set_resolution(resolution[0], resolution[1])
        
        # Get actual resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video capture ready: {self.width}x{self.height} @ {self.fps:.1f} FPS")
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set capture resolution
        
        Args:
            width: Target width
            height: Target height
        
        Returns:
            True if successful
        """
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width != width or actual_height != height:
                logger.warning(
                    f"Requested {width}x{height}, got {actual_width}x{actual_height}"
                )
            
            self.width = actual_width
            self.height = actual_height
            return True
        
        return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
        
        return ret, frame
    
    def release(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()
            logger.info("Video capture released")
    
    def get_position(self) -> float:
        """
        Get current position in video (0.0 to 1.0)
        
        Returns:
            Position ratio (0.0 = start, 1.0 = end)
        """
        if self.total_frames > 0:
            return self.frame_count / self.total_frames
        return 0.0
    
    def is_opened(self) -> bool:
        """Check if capture is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __del__(self):
        """Destructor"""
        self.release()


class VideoWriter:
    """Video writer utility"""
    
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = "mp4v"
    ):
        """
        Initialize video writer
        
        Args:
            output_path: Output video file path
            fps: Frames per second
            frame_size: Frame size (width, height)
            codec: Video codec (default: mp4v)
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_path}")
        
        logger.info(f"Video writer initialized: {output_path} ({frame_size[0]}x{frame_size[1]} @ {fps} FPS)")
    
    def write(self, frame: np.ndarray):
        """Write frame to video"""
        if self.writer:
            self.writer.write(frame)
    
    def release(self):
        """Release video writer"""
        if self.writer:
            self.writer.release()
            logger.info(f"Video saved: {self.output_path}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __del__(self):
        """Destructor"""
        self.release()
