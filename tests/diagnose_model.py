"""
Diagnose RetinaFace model output format
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("\n" + "="*60)
    print("RETINAFACE MODEL DIAGNOSTIC")
    print("="*60)
    
    model_path = Path("models/retinaface_resnet50.onnx")
    
    if not model_path.exists():
        print(f"[X] Model not found: {model_path}")
        return 1
    
    print(f"\n1. Loading model...")
    session = ort.InferenceSession(
        str(model_path),
        providers=['DmlExecutionProvider', 'CPUExecutionProvider']
    )
    print(f"[OK] Model loaded")
    
    print(f"\n2. Model Information:")
    print(f"   Inputs:")
    for inp in session.get_inputs():
        print(f"      Name: {inp.name}")
        print(f"      Shape: {inp.shape}")
        print(f"      Type: {inp.type}")
    
    print(f"\n   Outputs:")
    for out in session.get_outputs():
        print(f"      Name: {out.name}")
        print(f"      Shape: {out.shape}")
        print(f"      Type: {out.type}")
    
    print(f"\n3. Creating test input...")
    test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    
    print(f"[OK] Test input created: {test_input.shape}")
    
    print(f"\n4. Running inference...")
    outputs = session.run(output_names, {input_name: test_input})
    print(f"[OK] Inference complete")
    
    print(f"\n5. Output Analysis:")
    for i, (output, name) in enumerate(zip(outputs, output_names)):
        print(f"\n   Output {i} ({name}):")
        print(f"      Shape: {output.shape}")
        print(f"      Dtype: {output.dtype}")
        print(f"      Min: {output.min():.6f}")
        print(f"      Max: {output.max():.6f}")
        print(f"      Mean: {output.mean():.6f}")
        
        if output.ndim >= 2:
            print(f"      Sample (first detection):")
            if output.shape[0] > 0:
                if output.ndim == 3:
                    sample = output[0, 0, :]
                elif output.ndim == 2:
                    sample = output[0, :]
                else:
                    sample = output.flatten()[:20]
                print(f"         {sample[:min(20, len(sample))]}")
    
    print(f"\n6. Interpretation:")
    print(f"\n   Based on output shapes, this model likely uses:")
    
    if len(outputs) == 1:
        shape = outputs[0].shape
        if len(shape) == 3:
            print(f"      Format: Single output with shape {shape}")
            print(f"      Batch size: {shape[0]}")
            print(f"      Max detections: {shape[1]}")
            print(f"      Feature size: {shape[2]}")
            print(f"\n      Likely format:")
            print(f"         [0:4]   - BBox (x1, y1, x2, y2)")
            print(f"         [4]     - Confidence")
            print(f"         [5:15]  - Landmarks (5 points x 2 coords)")
        elif len(shape) == 2:
            print(f"      Format: Single output with shape {shape}")
            print(f"      Detections: {shape[0]}")
            print(f"      Feature size: {shape[1]}")
    else:
        print(f"      Format: Multiple outputs ({len(outputs)} outputs)")
        for i, out in enumerate(outputs):
            print(f"         Output {i}: {out.shape}")
    
    print("\n" + "="*60)
    print("[OK] DIAGNOSTIC COMPLETE")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
