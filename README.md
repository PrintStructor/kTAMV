<p align="center">
  <h1 align="center">kTAMV - Klipper Tool Alignment (using) Machine Vision</h1>
  <h3 align="center">Enhanced Fork with Improved Nozzle Detection</h3>
</p>

> **This is a fork of [TypQxQ/kTAMV](https://github.com/TypQxQ/kTAMV) with significantly improved nozzle detection accuracy and performance.**

---

## What's Changed in This Fork

### Completely Rewritten Detection Algorithm

| Feature | Original | This Fork |
|---------|----------|-----------|
| **Detection Method** | SimpleBlobDetector | 3-Stage Radial Symmetry + Ellipse Fitting |
| **Precision** | Pixel-level | Sub-pixel accuracy |
| **Stability** | Jumps between frames | Temporal smoothing (20 frames) |
| **Search Area** | Fixed area parameters | Full image, no limitations |

### The 3-Stage Detection Pipeline

1. **Stage 1 - Coarse Detection**
   - Full image analysis
   - Gradient-weighted voting at radii [30, 50, 70]
   - Finds approximate center of concentric circles

2. **Stage 2 - Fine Detection**
   - 80px ROI around Stage 1 result
   - More radii [15, 25, 35, 45, 55, 65] for precision
   - Gradient magnitude weighting

3. **Stage 3 - Ultra-Fine (Ellipse Fitting)**
   - 50px ROI around Stage 2 result
   - Canny edge detection for clean contours
   - `cv2.fitEllipse()` for sub-pixel center coordinates

### Performance Improvements

| Setting | Original | This Fork |
|---------|----------|-----------|
| Preview FPS | 2 | **15** |
| Resolution | 640×480 | **1280×720** |
| Centering Accuracy | ~1 pixel | **< 0.01mm (10µm)** |

### New Features

- **Temporal Smoothing**: Median filter over 20 frames for stable detection
- **Smart History Reset**: Detects toolhead movement (>40px jump) and resets history
- **Strict Outlier Rejection**: Only measurements within 15px of median are accepted
- **CLAHE Enhancement**: Adaptive contrast enhancement for better edge detection
- **Gradient Weighting**: Stronger edges get more voting power (quadratic weighting)

---

## Installation

### Fresh Install
```bash
cd ~
git clone https://github.com/PrintStructor/kTAMV.git
bash ~/kTAMV/install.sh
```

### Upgrade from Original kTAMV
```bash
cd ~
mv kTAMV kTAMV.backup
git clone https://github.com/PrintStructor/kTAMV.git
sudo systemctl restart ktamv
```

Your `printer.cfg` and `crowsnest.conf` settings remain unchanged - only the detection code is updated.

---

## Changed Files

```
server/ktamv_server_dm.py  - Main change (detection algorithm)
server/ktamv_server.py     - FPS + resolution settings
extension/ktamv.py         - Minor adjustments
extension/ktamv_utl.py     - Minor adjustments
server/ktamv_server_io.py  - Minor adjustments
```

---

## Why This Fork?

The original kTAMV uses OpenCV's `SimpleBlobDetector` which:
- Finds dark "blobs" rather than geometric circle centers
- Uses fixed area parameters that don't work for all nozzle sizes
- Has no temporal smoothing, causing detection to jump between frames
- Only provides integer pixel coordinates

This fork uses **Radial Symmetry Detection** which:
- Finds the exact geometric center of all concentric circles (the nozzle opening)
- Works regardless of nozzle size
- Provides sub-pixel precision through ellipse fitting
- Includes robust temporal smoothing for stable, consistent detection

---

## Original Documentation

For general usage, configuration, and commands, see the original documentation below.

---

## Commands

- `KTAMV_CALIB_CAMERA` - Calibrate camera mm/pixel ratio
- `KTAMV_FIND_NOZZLE_CENTER` - Center nozzle in camera view
- `KTAMV_SET_ORIGIN` - Set current position as reference
- `KTAMV_GET_OFFSET` - Measure offset from origin
- `KTAMV_MOVE_TO_ORIGIN` - Move to saved origin position
- `KTAMV_SIMPLE_NOZZLE_POSITION` - Check if nozzle is detected
- `KTAMV_START_PREVIEW` / `KTAMV_STOP_PREVIEW` - Control preview mode

---

## Configuration

```yaml
[ktamv]
nozzle_cam_url: http://localhost/webcam2/snapshot?max_delay=0
server_url: http://localhost:8085
move_speed: 1800
send_frame_to_cloud: false
detection_tolerance: 0
```

---

## Credits

- **Original kTAMV**: [TypQxQ/kTAMV](https://github.com/TypQxQ/kTAMV)
- **TAMV**: [HaythamB/TAMV](https://github.com/HaythamB/TAMV)
- **CVToolheadCalibration**: [cawmit/klipper_cv_toolhead_calibration](https://github.com/cawmit/klipper_cv_toolhead_calibration)

---

## License

Same license as the original kTAMV project.
