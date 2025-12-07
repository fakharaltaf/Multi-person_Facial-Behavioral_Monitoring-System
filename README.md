# Multi-Person Real-Time Facial Tracking & Behavioral Analysis System

## Project Overview
A real-time system for tracking and analyzing behavioral patterns of up to 30 people simultaneously using computer vision and deep learning.

## System Capabilities
- **Face Detection & Recognition**: RetinaFace + ArcFace/MagFace
- **Multi-Object Tracking**: StrongSORT/ByteTrack with appearance-based re-identification
- **Behavioral Analysis**:
  - Gaze direction estimation
  - Head pose estimation
  - Attentiveness classification
  - Yawn detection
- **Optimized for AMD RX 5700 XT** (ONNXRuntime + DirectML)

## Project Structure
```
├── config/              # Configuration files
├── models/              # Pre-trained model weights (ONNX)
├── src/
│   ├── detection/       # Face and person detection
│   ├── recognition/     # Face embedding and alignment
│   ├── tracking/        # Multi-object tracking
│   ├── behavior/        # Gaze, pose, yawn detection
│   ├── utils/           # Helper functions
│   └── visualization/   # UI and overlays
├── data/                # Test data and outputs
├── tests/               # Unit and integration tests
├── notebooks/           # Development and testing notebooks
└── requirements.txt     # Dependencies
```

## Development Approach
This project follows a staged implementation strategy:
1. **Stage 1**: Core detection (RetinaFace setup and testing)
2. **Stage 2**: Face recognition (ArcFace embedding pipeline)
3. **Stage 3**: Tracking system (StrongSORT/ByteTrack integration)
4. **Stage 4**: Behavioral modules (gaze, pose, yawn)
5. **Stage 5**: Integration and optimization

## Hardware Requirements
- **GPU**: AMD RX 5700 XT (DirectML support)
- **CPU**: 6-12 cores recommended
- **RAM**: Minimum 16GB
- **Camera**: 4K 30fps or 1080p with zoom

## Expected Performance
- **Resolution**: 1440p-4K
- **FPS**: 8-18 FPS (full pipeline)
- **Capacity**: 20-30 people simultaneously
- **ID Stability**: 80-95% reduction in ID switches

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Coming soon as each stage is completed.

## License
Educational Project - Deep Learning Course

## Authors
Season 7 Deep Learning Project Team
