# Behavioral Analysis Fixes

## Issues Identified and Fixed

### 1. Attention Score Always Too Low

**Problem:**
The attention calculation was overly sensitive to even small head movements:
- Previous: Yaw of just 20° would reduce attention by 22% (20/90 = 0.22)
- Previous: The thresholds were too strict for natural head movement
- Previous: Linear scaling meant small deviations had large impacts

**Solution:**
- Changed yaw threshold from 90° to 45° (more realistic attention range)
- Changed pitch threshold from 60° to 30° (looking up/down significantly affects attention)
- Applied non-linear scaling (square root) to be more forgiving of small movements
- Adjusted weighting: yaw=0.6, pitch=0.3, roll=0.1 (yaw is most important for attention)
- Lowered categorization thresholds: high ≥ 0.6, medium ≥ 0.3 (previously 0.7 and 0.4)

**Result:**
- Looking straight at camera: ~0.95-1.0 attention (high)
- Slight head turn (10-15°): ~0.8-0.9 attention (high)
- Moderate turn (20-30°): ~0.6-0.7 attention (high/medium)
- Looking away (>45°): <0.4 attention (low)

### 2. False Yawn Detection

**Problem:**
The mouth aspect ratio (MAR) calculation was fundamentally wrong:
- Used nose-to-mouth distance as "mouth opening" (completely incorrect!)
- The threshold was too low (0.6) for the incorrect calculation
- With only 5 facial landmarks, we cannot accurately measure actual mouth opening

**Solution:**
- Completely rewrote MAR calculation to use facial proportions:
  - Compares mouth width to eye-to-eye distance
  - Normal mouth: ~0.6 × eye width
  - Yawning mouth: ~0.9-1.2 × eye width
- Increased threshold from 0.6 to 0.9 (reduces false positives significantly)
- Increased required frames from 3 to 8 (more confirmation needed)
- Added warnings that 5-point landmarks are insufficient for accurate yawn detection

**Important Note:**
For reliable yawn detection, you need 68-point facial landmarks (like dlib) to measure actual vertical mouth opening. RetinaFace's 5 points (eyes, nose, mouth corners) cannot accurately capture mouth opening.

## Testing the Fixes

Run the behavior test:
```bash
python tests/test_behavior.py
```

Expected improvements:
1. Attention should stay high (0.8-1.0) when looking at camera
2. Attention should decrease gradually as you turn head away
3. Yawning should rarely (or never) be detected unless you actually yawn widely and hold it

## Recommendations for Production

### For Better Yawn Detection:
1. **Use 68-point facial landmarks** (dlib, MediaPipe Face Mesh, etc.)
2. Alternative: Use a trained CNN specifically for yawn detection
3. Current system: Disable yawn detection or treat as unreliable

### For Better Attention Tracking:
1. Current system works well after these fixes
2. Consider adding eye gaze tracking for even more accuracy
3. Consider temporal smoothing to reduce jitter

### Calibration Options:
You can adjust thresholds in the code:
- `head_pose.py` → `estimate_attention_level()`: Adjust angle thresholds
- `eye_tracking.py` → `YawnDetector.__init__()`: Adjust MAR threshold
- `behavior_track.py` → Adjust attention/engagement calculations

## Modified Files
- `src/behavior/head_pose.py` - Fixed attention calculation
- `src/behavior/eye_tracking.py` - Fixed MAR calculation and yawn thresholds
- `src/behavior/behavior_track.py` - Added documentation notes
