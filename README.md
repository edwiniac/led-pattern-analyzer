# LED Pattern Analyzer

Analyze LED patterns from video. Detects colors, timing, and patterns (blink/fade/solid) from smart lock LEDs and similar devices.

![Example](https://img.shields.io/badge/Python-3.8+-blue.svg)

## Quick Start

```bash
# Install dependencies
pip install opencv-python numpy

# Analyze a video (minimal output)
python led_analyzer.py video.mp4 --minimal --quiet

# Output:
# GREEN×5(137ms) → GREEN(512ms) → RED(307ms)
```

## Installation

```bash
git clone https://github.com/edwiniac/led-pattern-analyzer.git
cd led-pattern-analyzer
pip install opencv-python numpy
```

## Usage

### Basic Analysis
```bash
python led_analyzer.py video.mp4
```

### Minimal JSON Output (Recommended)
```bash
python led_analyzer.py video.mp4 --minimal -o result.json
```

**Output format:**
```json
{
  "sequence": ["green", "green", "green", "red"],
  "pattern": "GREEN×3(137ms) → RED(307ms)",
  "duration_sec": 7.72,
  "events": [
    {"t": 1.57, "color": "green", "ms": 137},
    {"t": 2.05, "color": "green", "ms": 137},
    {"t": 2.46, "color": "green", "ms": 137},
    {"t": 5.12, "color": "red", "ms": 307}
  ]
}
```

### Quick Pattern Check
```bash
python led_analyzer.py video.mp4 --quiet --minimal
# Just prints: GREEN×3(137ms) → RED(307ms)
```

### Create Annotated Video
```bash
python led_analyzer.py video.mp4 --annotate
# Creates video_annotated.mp4 with color labels
```

## Options

| Flag | Description |
|------|-------------|
| `-m, --minimal` | Output minimal JSON (easy to parse) |
| `-q, --quiet` | Suppress detailed output, just show pattern |
| `-a, --annotate` | Create annotated video with LED labels |
| `-o, --output` | Output JSON path |
| `--roi x,y,w,h` | Manual region of interest |
| `--ai-detect` | Use AI (OWL-ViT) for auto LED detection |
| `--yolo-model PATH` | Use custom YOLO model for detection |
| `-t, --threshold` | Min bright pixels for LED on (default: 200) |

---

## 🤖 AI-Powered LED Detection

For new device types, use AI to auto-detect LED regions:

### Zero-shot Detection (No Training)
```bash
# Install dependencies
pip install transformers torch

# Use OWL-ViT for auto-detection
python led_analyzer.py video.mp4 --ai-detect --minimal
```

### Train Custom YOLO Model (Better Accuracy)

```bash
# 1. Setup dataset structure
python led_detector.py train --setup --data-dir my_led_dataset

# 2. Auto-annotate from existing videos (review & correct!)
python led_detector.py annotate video1.mp4 --prefix lock1 --data-dir my_led_dataset
python led_detector.py annotate video2.mp4 --prefix lock2 --data-dir my_led_dataset

# 3. Train YOLO
pip install ultralytics
python led_detector.py train --data-dir my_led_dataset --epochs 50

# 4. Use trained model
python led_analyzer.py video.mp4 --yolo-model runs/detect/led_detector/weights/best.pt
```

### Detection Methods

| Method | Speed | Accuracy | Training Needed |
|--------|-------|----------|-----------------|
| CV (default) | ⚡ Fast | Good for known devices | No |
| OWL-ViT | 🐢 Slow | Good zero-shot | No |
| YOLO (custom) | ⚡ Fast | Best | Yes (50+ images) |

### LED Detector CLI

```bash
# Detect LED regions in video
python led_detector.py detect video.mp4 --method hybrid --visualize

# Methods: cv, owlvit, yolo, hybrid
```

## How It Works

1. **Auto-detects LED region** by finding bright, saturated pixels across frames
2. **Filters out non-LED objects** (white cards, hands) using saturation thresholds
3. **Classifies colors** from HSV values (red, orange, yellow, green, cyan, blue, white)
4. **Detects patterns**: blink (instant), fade_in, fade_out, solid
5. **Groups events** into sequences for easier interpretation

## Supported Colors

| Color | Hue Range (OpenCV) |
|-------|-------------------|
| 🔴 Red | 0-22, 170-180 |
| 🟠 Orange | 22-35 |
| 🟡 Yellow | 35-50 |
| 🟢 Green | 50-105 |
| 🔵 Blue | 105-135 |
| 🟣 Magenta | 135-170 |
| ⚪ White | Low saturation |

## Example Output

**Full report:**
```
======================================================================
LED PATTERN ANALYSIS REPORT
======================================================================

📹 Video: sample.mp4
   Duration: 7.72 seconds
   FPS: 29.3

📊 SUMMARY
Total LED events: 7

🎨 Colors detected:
   green     :   6 events, total 1.26s
   red       :   1 events, total 0.31s

📅 FORMATTED TIMELINE
→ GREEN solid (205ms) ×5
→ GREEN solid (512ms)
→ RED solid (307ms)
```

## Tips

- **Best results**: Stable camera, good contrast between LED and background
- **Manual ROI**: If auto-detection fails, use `--roi x,y,w,h`
- **Threshold tuning**: Lower `--threshold` for dim LEDs, raise for noisy videos
- **White card issues**: The analyzer automatically filters out low-saturation objects

## License

MIT
