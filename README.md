# Multi-Person Real-Time Facial Behavioral Monitoring System

## Project Overview
A complete real-time system for tracking and analyzing behavioral patterns of multiple people simultaneously using computer vision and deep learning. **All stages fully implemented and operational.**

## ✅ Completed Features

### Face Detection & Recognition
- **RetinaFace ResNet50**: Multi-scale face detection with 5-point landmarks
- **ArcFace ResNet100**: 512-dimensional face embeddings
- **Face Database**: Person registration and recognition with similarity matching
- **Alignment**: Affine transformation-based face alignment

### Multi-Object Tracking
- **Kalman Filter**: Smooth motion prediction and tracking
- **Appearance-based Re-ID**: Embedding similarity for track association
- **Track Management**: Confirmed/tentative states, age-based pruning
- **ID Persistence**: Maintains identities across occlusions

### Behavioral Analysis
- **Head Pose Estimation**: 
  - **68-point dlib landmarks** (primary, accurate ±70° rotation)
  - 5-point RetinaFace landmarks (fallback)
  - Yaw, pitch, roll angles via PnP algorithm
- **Attention Scoring**: Multi-factor weighted calculation (0.0-1.0)
- **Gaze Direction**: 8-directional classification
- **Eye Tracking**: 
  - Eye Aspect Ratio (EAR) calculation
  - Blink detection and counting
  - Drowsiness detection
- **Yawn Detection**: 
  - Mouth Aspect Ratio (MAR) with 68-point landmarks
  - Temporal filtering to avoid false positives from speech
- **Engagement Classification**: Engaged/Moderate/Disengaged levels

### Visualization
- **Behavioral Overlays**: Real-time pose axes, attention scores, alerts
- **68-Point Landmarks**: Toggle-able facial landmark visualization
- **Metrics Panel**: Comprehensive statistics display
- **Color-coded Boxes**: Engagement level indication

## Project Structure
```
├── config/
│   ├── system_config.yaml           # Main system configuration
│   └── model_urls.yaml              # Model download references
├── models/
│   ├── retinaface_resnet50.onnx     # Face detector
│   ├── arcface_resnet100.onnx       # Face recognition
│   ├── shape_predictor_68_face_landmarks.dat  # Dlib landmarks
│   └── README.md
├── src/
│   ├── detection/
│   │   ├── retinaface.py            # Face detection implementation
│   │   ├── anchor_decoder.py        # Anchor-based detection
│   │   └── utils.py
│   ├── recognition/
│   │   ├── arcface.py               # Face embedding extraction
│   │   ├── alignment.py             # Face alignment
│   │   └── face_database.py         # Person database
│   ├── tracking/
│   │   ├── tracker.py               # Multi-object tracker
│   │   ├── track.py                 # Track class with Kalman filter
│   │   └── visualization.py         # Tracking visualization
│   ├── behavior/
│   │   ├── behavior_tracker.py      # Behavioral analysis tracker
│   │   ├── behavior_track.py        # Track with behavioral state
│   │   ├── head_pose.py             # 5-point pose estimation
│   │   ├── dlib_head_pose.py        # 68-point pose estimation (dlib)
│   │   ├── eye_tracking.py          # Blink/drowsiness/yawn detection
│   │   └── visualization.py         # Behavioral overlays
│   ├── utils/
│   │   ├── config_loader.py         # YAML configuration
│   │   ├── logger.py                # Logging utilities
│   │   ├── video_capture.py         # Video I/O
│   │   ├── download_models.py       # Model downloader
│   │   └── verify_setup.py          # Setup verification
│   └── visualization/               # Additional visualization tools
├── data/
│   ├── face_database/               # Registered faces
│   └── outputs/                     # Results and logs
├── tests/
│   ├── test_behavior.py             # Full system integration test
│   ├── test_detection.py            # Face detection test
│   ├── test_recognition.py          # Face recognition test
│   ├── test_tracking.py             # Tracking test
│   ├── test_dlib_pose.py            # 68-point landmark test
│   ├── test_yawn_simple.py          # Yawn detection test
│   └── debug_yawn.py                # MAR value debugging
└── requirements.txt

## System Performance

### Hardware Optimized For
- **GPU**: AMD RX 5700 XT with DirectML acceleration
- **CPU**: Multi-core processing support
- **RAM**: 8-16GB recommended

### Achieved Metrics
- **Resolution**: 720p-1080p
- **FPS**: 15-17 FPS (full pipeline with dlib)
- **Capacity**: 30+ people simultaneously
- **Accuracy**:
  - Face Detection: 95%+ recall
  - Face Recognition: 0.75 similarity threshold (minimal false matches)
  - Head Pose: ±70° detection range with dlib
  - Attention Scoring: 0.95-1.0 for frontal, gradual decrease
  - Yawn Detection: 0.5 MAR threshold, 10-frame temporal filter

## Quick Start

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download models (dlib shape predictor downloads automatically)
python src/utils/download_models.py --all
```

### 2. Run Full System
```bash
# Complete behavioral analysis with all features
python tests/test_behavior.py

# Controls:
#   'r' - Register current faces to database
#   'p' - Toggle pose axes visualization
#   'd' - Toggle detailed behavioral info
#   'l' - Toggle 68-point landmarks
#   's' - Show statistics
#   'e' - Show engagement report
#   'c' - Clear all tracks
#   'q' - Quit
```

### 3. Individual Module Tests
```bash
# Face detection only
python tests/test_detection.py

# Face recognition and database
python tests/test_recognition.py

# Tracking without behavioral analysis
python tests/test_tracking.py

# 68-point landmark pose estimation
python tests/test_dlib_pose.py

# Yawn detection calibration
python tests/debug_yawn.py
python tests/test_yawn_simple.py
```

## Configuration

Key settings in `config/system_config.yaml`:

```yaml
detection:
  confidence_threshold: 0.5        # Face detection confidence
  nms_threshold: 0.25             # Non-maximum suppression

recognition:
  similarity_threshold: 0.75      # Face matching threshold

tracking:
  max_age: 30                     # Frames before track deletion
  min_hits: 3                     # Confirmations needed
  iou_threshold: 0.3              # IoU for matching
  embedding_threshold: 0.5        # Appearance similarity

behavior:
  use_dlib: true                  # Enable 68-point landmarks
  yawn_threshold: 0.5             # MAR threshold for yawns
  yawn_frames: 10                 # Consecutive frames required
  attention_thresholds:
    high: 0.5                     # Attention > 0.5 = high
    medium: 0.25                  # 0.25-0.5 = medium
```

## Key Improvements

### Attention Scoring
- **Problem**: Original 5-point landmarks gave unrealistic angles (±2-15°)
- **Solution**: Implemented dlib 68-point landmarks
- **Result**: Accurate ±70° rotation detection, realistic attention scores

### Yawn Detection
- **Problem**: False positives with 5-point MAR calculation
- **Solution**: 
  - Proper 68-point MAR using mouth landmarks 48-67
  - Lower threshold (0.5 vs 0.9) with longer duration (10 frames)
  - Temporal filtering to distinguish from speech
- **Result**: Reliable yawn detection without false positives from talking

### Duplicate Detection
- **Problem**: Multiple overlapping bounding boxes
- **Solution**: Enhanced NMS with containment and distance checks
- **Result**: Single detection per face

### Face Recognition
- **Problem**: Everyone detected as same person
- **Solution**: Increased similarity threshold from 0.6 to 0.75
- **Result**: Minimal false matches, reliable identity tracking

## Documentation

- **README.md** (this file): Project overview
- **QUICKSTART.md**: Setup and usage guide
- **STATUS.md**: Implementation status
- **DLIB_INTEGRATION_COMPLETE.md**: 68-point landmark details
- **models/README.md**: Model information

## Dependencies

### Core
- Python 3.8+
- opencv-python
- numpy
- onnxruntime-directml (AMD GPU)

### Computer Vision
- dlib (68-point facial landmarks)
- scipy, scikit-learn
- filterpy (Kalman filtering)

### Utilities
- PyYAML, tqdm, psutil
- matplotlib, pandas

## License
Educational Project - Deep Learning Course

## Authors
Deep Learning Season 7 Project Team
