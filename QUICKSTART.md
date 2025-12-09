# Quick Start Guide

## Prerequisites
- **Python**: 3.8 or higher
- **GPU**: AMD RX 5700 XT or compatible (DirectML support)
- **RAM**: 8GB minimum, 16GB recommended
- **Webcam**: 720p or higher resolution

## Step 1: Environment Setup

### Install Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install all requirements
pip install -r requirements.txt
```

**Key Dependencies Installed:**
- `onnxruntime-directml` - AMD GPU acceleration
- `opencv-python` - Computer vision
- `dlib` - 68-point facial landmarks
- `numpy`, `scipy`, `filterpy` - Math and filtering
- `pyyaml`, `tqdm` - Configuration and progress

### Verify Installation

```powershell
python src/utils/verify_setup.py
```

This checks:
- Python version compatibility
- All required packages installed
- ONNXRuntime DirectML support
- GPU availability and recognition
- Project structure integrity

## Step 2: Download Models

### Automatic Download (Recommended)

```powershell
# Download all required models
python src/utils/download_models.py --all
```

This downloads:
- `retinaface_resnet50.onnx` - Face detection
- `arcface_resnet100.onnx` - Face recognition
- `shape_predictor_68_face_landmarks.dat` - Dlib landmarks (downloads automatically when needed)

### Manual Download (if needed)

If automatic download fails, see `models/README.md` for:
- Direct download links
- Manual installation instructions
- Model conversion guides

## Step 3: Run the System

### Full Behavioral Analysis (Recommended)

```powershell
# Launch complete system with all features
python tests/test_behavior.py
```

**Features active:**
- Real-time face detection and recognition
- Multi-person tracking with ID persistence
- 68-point facial landmark detection
- Head pose estimation (yaw, pitch, roll)
- Attention scoring and engagement classification
- Blink and drowsiness detection
- Yawn detection with temporal filtering
- Visual overlays and metrics panel

**Interactive Controls:**
- **'r'** - Register current faces to database (save identities)
- **'p'** - Toggle pose axes visualization
- **'d'** - Toggle detailed behavioral info overlay
- **'l'** - Toggle 68-point facial landmarks display
- **'s'** - Show tracking statistics (console)
- **'e'** - Show engagement report (console)
- **'c'** - Clear all active tracks
- **'q'** - Quit application

### Individual Module Tests

```powershell
# Face detection only
python tests/test_detection.py

# Face recognition and person matching
python tests/test_recognition.py

# Multi-object tracking
python tests/test_tracking.py

# 68-point landmark pose estimation
python tests/test_dlib_pose.py

# Yawn detection with MAR visualization
python tests/test_yawn_simple.py

# Yawn threshold calibration tool
python tests/debug_yawn.py
```

## Step 4: Register People

To enable face recognition:

1. Run `python tests/test_behavior.py`
2. Position person in front of camera (frontal view works best)
3. Wait for stable detection (green bounding box)
4. Press **'r'** to enter registration mode
5. Enter person's name when prompted
6. Person will be recognized in future sessions

**Registered faces stored in:** `data/face_database/metadata.json`

## Step 5: Configuration (Optional)

Edit `config/system_config.yaml` to customize:

### Detection Settings
```yaml
detection:
  confidence_threshold: 0.5    # Lower = more detections
  nms_threshold: 0.25         # Lower = fewer duplicates
```

### Recognition Settings
```yaml
recognition:
  similarity_threshold: 0.75  # Higher = stricter matching
```

### Behavioral Analysis
```yaml
behavior:
  use_dlib: true              # Enable 68-point landmarks
  yawn_threshold: 0.5         # MAR threshold for yawns
  yawn_frames: 10             # Consecutive frames required
```

### Performance Tuning
```yaml
system:
  target_fps: 15              # Frame rate target
  max_people: 30              # Maximum tracked people
```

## Common Issues & Solutions

### "No faces detected"
- **Cause**: Low lighting or face too small
- **Solution**: 
  - Improve lighting
  - Move closer to camera
  - Lower `confidence_threshold` in config (e.g., 0.4)

### "Everyone detected as same person"
- **Cause**: Similarity threshold too low
- **Solution**: 
  - Increase `similarity_threshold` to 0.80 or higher
  - Re-register people with better quality images
  - Ensure frontal face during registration

### Duplicate bounding boxes
- **Cause**: NMS threshold too high
- **Solution**: Already fixed with enhanced NMS (threshold 0.25)
- If still occurs, lower `nms_threshold` to 0.20

### False yawn detection
- **Cause**: Talking triggering yawn detector
- **Solution**: Already fixed with temporal filtering (10 frames)
- If still occurs, increase `yawn_frames` to 12-15

### DirectML Not Working

```powershell
# Verify DirectML support
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Should include 'DmlExecutionProvider'
```

If not listed:
```powershell
pip uninstall onnxruntime onnxruntime-directml
pip install onnxruntime-directml
```

### Low FPS (< 10 FPS)

**Quick fixes:**
```yaml
# In config/system_config.yaml
system:
  target_fps: 10          # Lower target
  
detection:
  confidence_threshold: 0.6  # Fewer false detections
  
behavior:
  use_dlib: false         # Disable 68-point (faster but less accurate)
```

**Or use smaller input resolution:**
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### Memory Issues

If system runs out of memory:
- Reduce `max_people` to 15-20
- Close other applications
- Disable landmark visualization ('l' key)

### Dlib fails to load

```powershell
# Reinstall dlib
pip uninstall dlib
pip install dlib
```

If that fails, system automatically falls back to 5-point landmarks

## Feature Reference

### Attention Scoring (0.0 - 1.0)
- **1.00**: Perfect frontal attention
- **0.70-0.95**: High attention (looking at screen)
- **0.50-0.70**: Medium attention (slight angle)
- **0.25-0.50**: Low attention (looking away)
- **< 0.25**: Very low attention (not engaged)

### Engagement Levels
- **Engaged**: Attention > 0.5, stable tracking
- **Moderate**: Attention 0.25-0.5, some distraction
- **Disengaged**: Attention < 0.25, not focused

### Head Pose Ranges
- **Yaw**: ±70° detection range (left/right rotation)
- **Pitch**: ±40° typical range (up/down tilt)
- **Roll**: ±50° typical range (head tilt)

### Behavioral Thresholds
- **Blink**: EAR < 0.21 for 3+ frames
- **Drowsiness**: Average EAR < 0.23 over 30 frames
- **Yawn**: MAR > 0.5 for 10+ consecutive frames

## System Architecture

```
Input (Webcam/Video)
  ↓
RetinaFace Detection (5-point landmarks)
  ↓
ArcFace Recognition (512-d embeddings)
  ↓
Multi-Object Tracker (Kalman filter + appearance)
  ↓
Dlib 68-Point Landmarks (for accurate pose)
  ↓
Behavioral Analysis:
  • Head Pose (PnP algorithm)
  • Attention Scoring
  • Eye Tracking (EAR)
  • Yawn Detection (MAR)
  • Engagement Classification
  ↓
Visualization & Metrics
  ↓
Output (Display + Database)
```

## Next Steps

1. **Explore settings**: Experiment with thresholds in `config/system_config.yaml`
2. **Build database**: Register family, friends, or team members
3. **Analyze behavior**: Use 's' and 'e' keys for real-time statistics
4. **Optimize performance**: Adjust FPS and resolution for your hardware
5. **Review documentation**: See `DLIB_INTEGRATION_COMPLETE.md` for technical details

## Support

For issues or questions:
1. Check `STATUS.md` for implementation details
2. Review test scripts in `tests/` directory
3. Examine configuration in `config/system_config.yaml`
4. See model documentation in `models/README.md`

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
