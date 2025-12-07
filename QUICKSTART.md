# Quick Start Guide

## Step 1: Environment Setup

### Install Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Verify Installation

```powershell
python src/utils/verify_setup.py
```

This will check:
- Python version
- Required packages
- ONNXRuntime DirectML support
- GPU availability

## Step 2: Download Models

```powershell
# List available models
python src/utils/download_models.py --list

# Download all models (recommended for first setup)
python src/utils/download_models.py --all

# Or download specific models
python src/utils/download_models.py --model detection/retinaface_resnet50
python src/utils/download_models.py --model recognition/arcface_resnet100
```

**Note**: Some models may require manual download. See `models/README.md` for details.

## Step 3: Test Setup

### Test with Single Image

```powershell
# Test face detection
python tests/test_detection.py --image data/test_images/sample.jpg

# Test with visualization
python tests/test_detection.py --image data/test_images/sample.jpg --visualize
```

### Test with Video

```powershell
# Test with webcam
python tests/test_detection.py --source webcam

# Test with video file
python tests/test_detection.py --video data/test_videos/sample.mp4
```

## Step 4: Configuration

Edit `config/system_config.yaml` to customize:

- Detection thresholds
- Tracking parameters
- Behavioral analysis settings
- Input/output settings
- Performance optimization

## Step 5: Run Full Pipeline

```powershell
# Run full tracking system (when implemented)
python src/main.py --config config/system_config.yaml
```

## Development Workflow

We're following a staged implementation:

1. **Stage 1 (CURRENT)**: Face Detection - RetinaFace implementation
2. **Stage 2**: Face Recognition - ArcFace embedding
3. **Stage 3**: Multi-Object Tracking - StrongSORT/ByteTrack
4. **Stage 4**: Behavioral Analysis - Gaze, pose, yawn
5. **Stage 5**: Integration & Optimization

See `DEVELOPMENT_ROADMAP.md` for detailed plan.

## Testing Each Stage

After implementing each stage:

```powershell
# Run stage-specific tests
python tests/test_stage1_detection.py
python tests/test_stage2_recognition.py
python tests/test_stage3_tracking.py
python tests/test_stage4_behavior.py
python tests/test_stage5_integration.py
```

## Common Issues

### DirectML Not Working

```powershell
# Verify DirectML installation
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Should include 'DmlExecutionProvider'
```

### Memory Issues

- Reduce `max_people` in config
- Lower `detection_resolution`
- Reduce `max_batch_size`

### Low FPS

- Use mobile variant: `retinaface_mobile025`
- Enable `detection_interval` (skip frames)
- Reduce input resolution
- Disable unnecessary features

## Project Structure

```
├── config/              # Configuration files
├── models/              # ONNX models (download required)
├── src/
│   ├── detection/       # Stage 1: Face detection
│   ├── recognition/     # Stage 2: Face embedding
│   ├── tracking/        # Stage 3: Multi-object tracking
│   ├── behavior/        # Stage 4: Behavioral analysis
│   ├── utils/           # Utilities (logging, video, config)
│   └── visualization/   # Display and rendering
├── data/
│   ├── test_images/     # Test images
│   ├── test_videos/     # Test videos
│   └── outputs/         # Results and logs
├── tests/               # Test scripts
└── notebooks/           # Jupyter notebooks for experimentation
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Download models
3. ✅ Verify setup
4. 📍 **YOU ARE HERE** - Start Stage 1: Face Detection
5. ⏳ Test detection on sample images
6. ⏳ Implement remaining stages

## Getting Help

- Check `DEVELOPMENT_ROADMAP.md` for implementation details
- Review configuration in `config/system_config.yaml`
- See model info in `config/model_urls.yaml`
- Read module documentation in respective `README.md` files

## Performance Expectations (RX 5700 XT)

- **Detection**: 15-25 FPS @ 1080p
- **Full Pipeline**: 10-18 FPS @ 1440p
- **Max People**: 20-30 simultaneously
- **ID Stability**: 80-95% (low ID switches)

Happy coding! 🚀
