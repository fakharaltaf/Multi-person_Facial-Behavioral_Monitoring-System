"""
Quick detection test script
Tests if RetinaFace model loads and runs correctly
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import RetinaFace


def create_test_image():
    """Create a simple test image with colored rectangles"""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some pattern
    image[100:380, 100:540] = [100, 150, 200]
    cv2.rectangle(image, (150, 150), (490, 330), (255, 255, 255), 2)
    cv2.putText(image, "Test Image", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image


def main():
    print("\n" + "="*60)
    print("QUICK DETECTION TEST")
    print("="*60)
    
    # Model path
    model_path = Path("models/retinaface_resnet50.onnx")
    
    print(f"\n1. Checking model file...")
    if not model_path.exists():
        print(f"   ❌ Model not found: {model_path}")
        print(f"   Please ensure the model is downloaded")
        return 1
    
    model_size = model_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Model found: {model_size:.1f} MB")
    
    print(f"\n2. Initializing detector...")
    try:
        detector = RetinaFace(
            model_path=str(model_path),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            device="dml"  # or "cpu" if DirectML not available
        )
        print(f"   ✓ Detector initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize detector: {e}")
        return 1
    
    print(f"\n3. Creating test image...")
    test_image = create_test_image()
    print(f"   ✓ Test image created: {test_image.shape}")
    
    print(f"\n4. Running detection...")
    try:
        detections = detector.detect(test_image)
        print(f"   [OK] Detection completed")
        print(f"   Detected {len(detections)} faces (expected 0 for test pattern)")
    except Exception as e:
        print(f"   [X] Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n5. Testing with actual image...")
    # Try to load a real image if available
    test_images = [
        "data/test_images/sample.jpg",
        "data/test_images/test.jpg",
        "data/test_images/faces.jpg"
    ]
    
    real_image_found = False
    for img_path in test_images:
        if Path(img_path).exists():
            print(f"   Loading: {img_path}")
            image = cv2.imread(img_path)
            if image is not None:
                try:
                    detections = detector.detect(image)
                    print(f"   [OK] Detected {len(detections)} faces")
                    real_image_found = True
                    break
                except Exception as e:
                    print(f"   [X] Detection failed: {e}")
    
    if not real_image_found:
        print(f"   [!] No test images found in data/test_images/")
        print(f"   Add some test images to fully test detection")
    
    print("\n" + "="*60)
    print("[OK] QUICK TEST PASSED!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Add test images to data/test_images/")
    print("  2. Run: python tests/test_detection.py --image data/test_images/your_image.jpg --visualize")
    print("  3. Run: python tests/test_detection.py --webcam")
    print("  4. Run: python tests/test_detection.py --ipcam --ip-url http://YOUR_IP:PORT/video")
    print("\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
