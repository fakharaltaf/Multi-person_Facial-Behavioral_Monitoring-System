# Models Directory

This directory stores pre-trained ONNX models used by the tracking system.

## Required Models

### Face Detection
- `retinaface_mobile025.onnx` - Lightweight RetinaFace variant
- `retinaface_resnet50.onnx` - Full accuracy RetinaFace variant

### Face Recognition
- `arcface_resnet100.onnx` - ArcFace embedding model (512-dim)
- `arcface_mobilefacenet.onnx` - Lightweight ArcFace variant

### Gaze Estimation
- `gaze360.onnx` - Gaze360 model for gaze direction

### Optional
- `yolov8n.onnx` - Person detection fallback
- `face_landmarks.onnx` - 68-point facial landmarks

## Download Models

Use the model downloader utility:

```bash
# List available models
python src/utils/download_models.py --list

# Download all models
python src/utils/download_models.py --all

# Download specific model
python src/utils/download_models.py --model detection/retinaface_resnet50
```

## Manual Download

Some models may require manual download. Check `config/model_urls.yaml` for sources.

**Important sources:**
- InsightFace Model Zoo: https://github.com/deepinsight/insightface/tree/master/model_zoo
- ONNX Model Zoo: https://github.com/onnx/models
- Hugging Face: https://huggingface.co/models

## Model Conversion

If ONNX versions are unavailable, you may need to convert PyTorch models:

```python
import torch
import onnx

# Example conversion
model = load_pytorch_model()
dummy_input = torch.randn(1, 3, 112, 112)

torch.onnx.export(
    model,
    dummy_input,
    "output.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

## Model Info

Each model should include:
- Input size
- Output format
- Preprocessing requirements
- Expected performance

See `config/model_urls.yaml` for detailed information.
