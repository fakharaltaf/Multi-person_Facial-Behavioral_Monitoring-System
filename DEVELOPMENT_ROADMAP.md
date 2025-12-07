# Development Roadmap

## Current Status: Stage 1 - Foundation Complete ✓

### Completed
- [x] Project structure created
- [x] Configuration system implemented
- [x] Logging utilities
- [x] Video capture abstraction
- [x] Documentation framework

---

## Stage 1: Core Face Detection (CURRENT FOCUS)

### Objectives
Implement and test RetinaFace detection system with ONNX on AMD GPU.

### Tasks
- [ ] Download and prepare RetinaFace ONNX models
  - [ ] Mobile0.25 variant (fast)
  - [ ] ResNet50 variant (accurate)
- [ ] Create ONNX inference wrapper with DirectML
- [ ] Implement preprocessing pipeline
- [ ] Implement postprocessing (NMS, landmark extraction)
- [ ] Create test script for single image
- [ ] Create test script for video stream
- [ ] Benchmark performance on RX 5700 XT
- [ ] Optimize batch processing
- [ ] Handle edge cases (no faces, many faces, small faces)

### Success Criteria
- Detect 20-30 faces reliably at 15+ FPS on 1080p
- Accurate bounding boxes with <5% false positives
- Stable landmark detection
- Documented performance metrics

### Testing Strategy
1. Test with single face image
2. Test with multi-face image (5, 10, 20, 30 people)
3. Test with video stream
4. Test with poor lighting
5. Test with occlusions
6. Measure FPS and accuracy

---

## Stage 2: Face Recognition & Alignment

### Objectives
Implement ArcFace embedding system with proper alignment.

### Tasks
- [ ] Download ArcFace ONNX models
- [ ] Implement face alignment using 5-point landmarks
- [ ] Create embedding extraction pipeline
- [ ] Implement embedding similarity computation
- [ ] Create face quality assessment module
- [ ] Test embedding consistency across frames
- [ ] Build embedding database structure
- [ ] Implement embedding smoothing (EMA)

### Success Criteria
- Consistent embeddings for same person (>0.85 similarity)
- Different people have low similarity (<0.5)
- Alignment works with varying poses
- Embeddings computed at 30+ FPS batched

---

## Stage 3: Multi-Object Tracking

### Objectives
Implement robust tracking with appearance-based re-identification.

### Tasks
- [ ] Implement Kalman filter for motion prediction
- [ ] Create IOU matching algorithm
- [ ] Integrate appearance (embedding) matching
- [ ] Implement track lifecycle management
- [ ] Create re-ID memory buffer
- [ ] Implement track association logic
- [ ] Add track smoothing and filtering
- [ ] Test with challenging scenarios (occlusions, crossings)

### Success Criteria
- ID switches <5% in controlled environment
- Tracks survive 2-3 second occlusions
- Correct re-identification after temporary disappearance
- Handles 30 people simultaneously

---

## Stage 4: Behavioral Analysis Modules

### 4a: Head Pose Estimation
- [ ] Implement 3D head model
- [ ] Create SolvePnP-based pose estimator
- [ ] Add temporal smoothing
- [ ] Test accuracy

### 4b: Gaze Estimation
- [ ] Download/convert Gaze360 model
- [ ] Implement gaze inference
- [ ] Create gaze direction classifier
- [ ] Add smoothing

### 4c: Eye Analysis
- [ ] Implement EAR (Eye Aspect Ratio) calculation
- [ ] Create blink detector
- [ ] Implement eye openness classifier
- [ ] Add temporal filtering

### 4d: Yawn Detection
- [ ] Implement MAR (Mouth Aspect Ratio) calculation
- [ ] Create yawn event detector
- [ ] Add duration tracking
- [ ] Test with real yawn examples

### 4e: Attentiveness Classification
- [ ] Combine all behavioral features
- [ ] Implement scoring system
- [ ] Add temporal windowing
- [ ] Create hysteresis mechanism
- [ ] Test accuracy vs ground truth

---

## Stage 5: Integration & Optimization

### Tasks
- [ ] Integrate all modules into main pipeline
- [ ] Implement multi-threading for parallel processing
- [ ] Optimize memory usage
- [ ] Add frame skipping for performance
- [ ] Create comprehensive visualization
- [ ] Implement data logging (CSV, JSON, SQLite)
- [ ] Add real-time dashboard
- [ ] Performance profiling and optimization
- [ ] Error handling and recovery
- [ ] Documentation and user guide

---

## Stage 6: Testing & Validation

### Tasks
- [ ] Create comprehensive test suite
- [ ] Record test videos with ground truth
- [ ] Measure ID switch rate
- [ ] Measure behavioral detection accuracy
- [ ] Stress test with 30+ people
- [ ] Test in various lighting conditions
- [ ] Test with different camera angles
- [ ] Performance benchmarking

---

## Stage 7: Optional Extensions

### Potential Features
- [ ] Emotion recognition module
- [ ] Age/gender estimation
- [ ] Speaking detection (audio + visual)
- [ ] Multi-camera support
- [ ] Web dashboard
- [ ] Real-time alerts
- [ ] Classroom analytics
- [ ] Automated attendance

---

## Development Guidelines

### Before Moving to Next Stage
1. ✅ All tasks completed
2. ✅ Tests passing
3. ✅ Performance meets targets
4. ✅ Code documented
5. ✅ Integration verified
6. ✅ Stakeholder approval

### Testing Philosophy
- Test after each component
- Use real-world test cases
- Measure quantitative metrics
- Document failures and edge cases
- Iterate until success criteria met

### Performance Targets
- **Overall FPS**: 10-18 FPS @ 1440p
- **Detection**: 15+ FPS
- **Recognition**: 30+ FPS (batched)
- **Tracking**: Minimal overhead
- **Behavioral**: 20+ FPS per module

---

## Next Steps

**Immediate Focus**: Stage 1 - Face Detection Implementation

1. Set up model download scripts
2. Implement RetinaFace ONNX inference
3. Test on sample images
4. Optimize for RX 5700 XT
5. Create comprehensive test suite

**Timeline Estimate**: 2-3 days per stage (7 stages = 14-21 days total)

---

## Notes

- Prioritize stability and accuracy over speed
- Test thoroughly at each stage
- Document all decisions and trade-offs
- Keep code modular and maintainable
- Regular performance profiling
- Iterative improvement based on testing
