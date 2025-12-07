"""
Test face detection module
Usage: python tests/test_detection.py --image path/to/image.jpg
"""

import sys
import argparse
from pathlib import Path
import cv2
import time
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import RetinaFace, draw_detections, filter_small_faces
from src.utils import get_config, setup_logger


def test_single_image(args):
    """Test detection on a single image"""
    print("\n" + "="*60)
    print("Testing Face Detection - Single Image")
    print("="*60)
    
    # Load configuration
    config = get_config()
    
    # Setup logger
    logger = setup_logger(level="INFO")
    
    # Initialize detector
    print(f"\nInitializing RetinaFace detector...")
    model_path = Path(config['detection']['model_path'])
    
    if not model_path.exists():
        print(f"[X] Model not found: {model_path}")
        print(f"Please ensure the model is in the models/ directory")
        return False
    
    detector = RetinaFace(
        model_path=str(model_path),
        confidence_threshold=config['detection']['confidence_threshold'],
        nms_threshold=config['detection']['nms_threshold'],
        device=config['system']['device']
    )
    
    print(f"[OK] Detector initialized")
    
    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"❌ Failed to load image")
        return False
    
    print(f"✓ Image loaded: {image.shape}")
    
    # Run detection
    print(f"\nRunning face detection...")
    start_time = time.time()
    detections = detector.detect(image)
    elapsed = time.time() - start_time
    
    print(f"✓ Detection complete in {elapsed*1000:.1f}ms")
    print(f"✓ Detected {len(detections)} faces")
    
    # Display results
    if len(detections) > 0:
        print(f"\nDetection Results:")
        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['confidence']
            has_landmarks = det['landmarks'] is not None
            print(f"  Face {i+1}:")
            print(f"    BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            print(f"    Confidence: {conf:.3f}")
            print(f"    Landmarks: {'Yes' if has_landmarks else 'No'}")
    
    # Filter small faces if requested
    if args.min_face_size:
        detections = filter_small_faces(detections, args.min_face_size)
        print(f"\n[OK] After filtering (min size {args.min_face_size}): {len(detections)} faces")
    
    # Visualize if requested
    if args.visualize:
        print(f"\nVisualizing results...")
        vis_image = draw_detections(
            image,
            detections,
            draw_landmarks=True,
            draw_confidence=True
        )
        
        # Display
        window_name = f"Detection Results ({len(detections)} faces)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, vis_image)
        
        print(f"[OK] Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis_image)
            print(f"[OK] Saved to: {output_path}")
    
    print("\n" + "="*60)
    print(f"[OK] Test completed successfully!")
    print("="*60 + "\n")
    
    return True


def test_webcam(args):
    """Test detection on webcam"""
    print("\n" + "="*60)
    print("Testing Face Detection - Webcam")
    print("="*60)
    
    # Load configuration
    try:
        config = get_config()
    except Exception as e:
        print(f"[X] Failed to load config: {e}")
        return False
    
    # Initialize detector
    print(f"\nInitializing RetinaFace detector...")
    model_path = Path(config['detection']['model_path'])
    
    if not model_path.exists():
        print(f"[X] Model not found: {model_path}")
        return False
    
    try:
        detector = RetinaFace(
            model_path=str(model_path),
            confidence_threshold=config['detection']['confidence_threshold'],
            nms_threshold=config['detection']['nms_threshold'],
            device=config['system']['device']
        )
        print(f"[OK] Detector initialized")
    except Exception as e:
        print(f"[X] Failed to initialize detector: {e}")
        return False
    
    # Open webcam with retries
    print(f"\nOpening camera {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)  # Use DirectShow on Windows
    
    if not cap.isOpened():
        print(f"[X] Failed to open camera {args.camera_id}")
        print(f"Trying alternative method...")
        cap = cv2.VideoCapture(args.camera_id)
        
    if not cap.isOpened():
        print(f"[X] Could not open camera. Please check:")
        print(f"  - Camera is connected")
        print(f"  - Camera permissions granted")
        print(f"  - No other app is using the camera")
        return False
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[OK] Camera opened: {actual_width}x{actual_height}")
    print(f"\nControls:")
    print(f"  'q' - Quit")
    print(f"  's' - Save screenshot")
    print(f"  '+' - Increase confidence threshold")
    print(f"  '-' - Decrease confidence threshold")
    print("")
    
    frame_times = []
    fps = 0
    confidence_threshold = detector.confidence_threshold
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Detect faces
            start_time = time.time()
            try:
                detections = detector.detect(frame)
            except Exception as e:
                print(f"[!] Detection error: {e}")
                detections = []
            
            elapsed = time.time() - start_time
            
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            avg_time = np.mean(frame_times) if frame_times else elapsed
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Draw results
            vis_frame = draw_detections(
                frame,
                detections,
                draw_landmarks=True,
                draw_confidence=True
            )
            
            # Draw stats overlay
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (5, 5), (400, 100), (0, 0, 0), -1)
            vis_frame = cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0)
            
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Faces: {len(detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Conf: {confidence_threshold:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("Face Detection - Webcam (Press Q to quit)", vis_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('s'):
                output_path = Path("data/outputs") / f"webcam_{int(time.time())}.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), vis_frame)
                print(f"[OK] Screenshot saved: {output_path}")
            elif key == ord('+') or key == ord('='):
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
                detector.confidence_threshold = confidence_threshold
                print(f"[OK] Confidence threshold: {confidence_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                confidence_threshold = max(0.1, confidence_threshold - 0.05)
                detector.confidence_threshold = confidence_threshold
                print(f"[OK] Confidence threshold: {confidence_threshold:.2f}")
    
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
    except Exception as e:
        print(f"\n[X] Error during capture: {e}")
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"\n[OK] Average FPS: {fps:.1f}")
    print("="*60 + "\n")
    
    return True


def test_ipcam(args):
    """Test detection on IP camera stream"""
    print("\n" + "="*60)
    print("Testing Face Detection - IP Camera")
    print("="*60)
    
    # Load configuration
    try:
        config = get_config()
    except Exception as e:
        print(f"[X] Failed to load config: {e}")
        return False
    
    # Initialize detector
    print(f"\nInitializing RetinaFace detector...")
    model_path = Path(config['detection']['model_path'])
    
    if not model_path.exists():
        print(f"[X] Model not found: {model_path}")
        return False
    
    try:
        detector = RetinaFace(
            model_path=str(model_path),
            confidence_threshold=config['detection']['confidence_threshold'],
            nms_threshold=config['detection']['nms_threshold'],
            device=config['system']['device']
        )
        print(f"[OK] Detector initialized")
    except Exception as e:
        print(f"[X] Failed to initialize detector: {e}")
        return False
    
    # Open IP camera stream
    ip_url = args.ip_url
    print(f"\nConnecting to IP camera: {ip_url}")
    print(f"This may take a few seconds...")
    
    cap = cv2.VideoCapture(ip_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    if not cap.isOpened():
        print(f"[X] Failed to connect to IP camera")
        print(f"\nCommon IP camera URL formats:")
        print(f"  RTSP: rtsp://username:password@ip:port/stream")
        print(f"  HTTP: http://ip:port/video")
        print(f"  DroidCam: http://ip:4747/video")
        print(f"  IP Webcam: http://ip:8080/video")
        return False
    
    print(f"[OK] Connected to IP camera")
    print(f"\nControls:")
    print(f"  'q' - Quit")
    print(f"  's' - Save screenshot")
    print(f"  '+' - Increase confidence threshold")
    print(f"  '-' - Decrease confidence threshold")
    print("")
    
    frame_times = []
    fps = 0
    frame_count = 0
    confidence_threshold = detector.confidence_threshold
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to read frame, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(ip_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            
            frame_count += 1
            
            # Detect faces
            start_time = time.time()
            try:
                detections = detector.detect(frame)
            except Exception as e:
                print(f"[!] Detection error: {e}")
                detections = []
            
            elapsed = time.time() - start_time
            
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            avg_time = np.mean(frame_times) if frame_times else elapsed
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Draw results
            vis_frame = draw_detections(
                frame,
                detections,
                draw_landmarks=True,
                draw_confidence=True
            )
            
            # Draw stats overlay
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (5, 5), (400, 130), (0, 0, 0), -1)
            vis_frame = cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0)
            
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Faces: {len(detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Conf: {confidence_threshold:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("Face Detection - IP Camera (Press Q to quit)", vis_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                output_path = Path("data/outputs") / f"ipcam_{int(time.time())}.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), vis_frame)
                print(f"[OK] Screenshot saved: {output_path}")
            elif key == ord('+') or key == ord('='):
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
                detector.confidence_threshold = confidence_threshold
                print(f"[OK] Confidence threshold: {confidence_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                confidence_threshold = max(0.1, confidence_threshold - 0.05)
                detector.confidence_threshold = confidence_threshold
                print(f"[OK] Confidence threshold: {confidence_threshold:.2f}")
    
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
    except Exception as e:
        print(f"\n[X] Error during capture: {e}")
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"\n[OK] Average FPS: {fps:.1f}")
    print(f"[OK] Processed {frame_count} frames")
    print("="*60 + "\n")
    
    return True


def test_video(args):
    """Test detection on video file"""
    print("\n" + "="*60)
    print("Testing Face Detection - Video File")
    print("="*60)
    
    # Load configuration
    config = get_config()
    
    # Initialize detector
    print(f"\nInitializing RetinaFace detector...")
    model_path = Path(config['detection']['model_path'])
    
    detector = RetinaFace(
        model_path=str(model_path),
        confidence_threshold=config['detection']['confidence_threshold'],
        nms_threshold=config['detection']['nms_threshold'],
        device=config['system']['device']
    )
    
    print(f"[OK] Detector initialized")
    
    # Open video
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[X] Video not found: {video_path}")
        return False
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"[X] Failed to open video")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[OK] Video opened: {total_frames} frames @ {fps} FPS")
    print(f"\nProcessing... Press 'q' to quit")
    
    frame_count = 0
    detection_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces
            start_time = time.time()
            detections = detector.detect(frame)
            elapsed = time.time() - start_time
            detection_times.append(elapsed)
            
            # Draw results
            vis_frame = draw_detections(frame, detections)
            
            # Draw progress
            progress = frame_count / total_frames * 100
            cv2.putText(
                vis_frame,
                f"Frame: {frame_count}/{total_frames} ({progress:.1f}%) | Faces: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display
            cv2.imshow("Face Detection - Video", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    avg_time = np.mean(detection_times)
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"\n[OK] Processed {frame_count} frames")
    print(f"[OK] Average detection time: {avg_time*1000:.1f}ms")
    print(f"[OK] Average FPS: {avg_fps:.1f}")
    print("="*60 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test face detection module")
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--video', type=str, help='Path to test video')
    parser.add_argument('--webcam', action='store_true', help='Test with webcam')
    parser.add_argument('--ipcam', action='store_true', help='Test with IP camera')
    parser.add_argument('--ip-url', type=str, help='IP camera URL (e.g., http://192.168.1.100:8080/video)')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--output', type=str, help='Output path for visualization')
    parser.add_argument('--min-face-size', type=int, help='Minimum face size to keep')
    
    args = parser.parse_args()
    
    # Determine test mode
    if args.image:
        success = test_single_image(args)
    elif args.video:
        success = test_video(args)
    elif args.webcam:
        success = test_webcam(args)
    elif args.ipcam:
        if not args.ip_url:
            print("[X] Please provide --ip-url for IP camera")
            print("\nExamples:")
            print("  --ip-url http://192.168.1.100:8080/video  # IP Webcam")
            print("  --ip-url http://192.168.1.100:4747/video  # DroidCam")
            print("  --ip-url rtsp://user:pass@192.168.1.100:554/stream  # RTSP")
            return 1
        success = test_ipcam(args)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python tests/test_detection.py --image data/test_images/sample.jpg --visualize")
        print("  python tests/test_detection.py --webcam")
        print("  python tests/test_detection.py --ipcam --ip-url http://192.168.1.100:8080/video")
        print("  python tests/test_detection.py --video data/test_videos/sample.mp4")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
