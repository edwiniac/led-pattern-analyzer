# LED Pattern Analyzer 💡

Detect and analyze LED colors, patterns, and timing from video footage. Designed for smart door lock LED indicators but works with any LED.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- 🎨 **Color Detection**: Red, orange, yellow, green, cyan, blue, magenta, white
- ⚡ **Pattern Recognition**: Blink, solid, fade-in, fade-out, fade
- ⏱️ **Timing Analysis**: Duration, rise/fall times, frequency
- 🔍 **Auto-Detection**: Automatically finds LED region in frame
- 📊 **JSON Export**: Structured data for further processing
- 🎬 **Annotated Video**: Visual verification of detection

## Installation

```bash
# Clone the repository
git clone https://github.com/edwiniac/led-pattern-analyzer.git
cd led-pattern-analyzer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Basic usage - auto-detects LED and analyzes
python led_analyzer.py video.mp4

# Output will be saved as video.json
```

## Usage

### Basic Analysis

```bash
python led_analyzer.py path/to/video.mp4
```

This will:
1. Auto-detect the LED region
2. Analyze every frame
3. Print a detailed report
4. Save results to `video.json`

### Options

```bash
python led_analyzer.py video.mp4 [OPTIONS]

Options:
  -o, --output PATH      Output JSON file path
  -a, --annotate         Create annotated video for verification
  -t, --threshold NUM    Min bright pixels for LED detection (default: 300)
  --roi X,Y,W,H          Manual ROI if auto-detection fails
```

### Examples

```bash
# Specify custom output path
python led_analyzer.py door_lock.mp4 -o results/analysis.json

# Create annotated video to verify detection
python led_analyzer.py door_lock.mp4 --annotate

# Adjust sensitivity for smaller LEDs
python led_analyzer.py door_lock.mp4 --threshold 150

# Manual ROI if auto-detection doesn't work
python led_analyzer.py door_lock.mp4 --roi 400,800,300,400
```

## Output Format

### Console Report

```
======================================================================
LED PATTERN ANALYSIS REPORT
======================================================================

📹 Video: sample.mp4
   Duration: 30.05 seconds
   FPS: 30.0

📊 SUMMARY
──────────────────────────────────────────────────────────────────────
Total LED events: 48

🎨 Colors detected:
   white     :  11 events, total 5.53s
   blue      :   6 events, total 2.47s
   red       :  11 events, total 1.13s

⚡ Patterns detected:
   blink     :  26 events
   solid     :  17 events
   fade_in   :   4 events
```

### JSON Structure

```json
{
  "video_path": "sample.mp4",
  "fps": 30.0,
  "duration_ms": 30046.68,
  "summary": {
    "total_events": 48,
    "colors": {"blue": 6, "white": 11, "red": 11},
    "patterns": {"blink": 26, "solid": 17, "fade_in": 4}
  },
  "events": [
    {
      "event_num": 1,
      "start_ms": 1499.0,
      "end_ms": 2898.1,
      "duration_ms": 1432.4,
      "color": "blue",
      "pattern": "fade_in",
      "rise_time_ms": 466.4,
      "fall_time_ms": null
    }
  ]
}
```

## How It Works

### Detection Method

The analyzer detects LEDs using **brightness and color**, not shape. This means it works with any LED form factor:

- ✅ Circular LEDs
- ✅ Pill/capsule shaped
- ✅ LED strips/bars
- ✅ Ring lights
- ✅ Multi-segment displays

**Detection criteria:**
1. **Brightness**: Pixels with value > 220 (in HSV color space)
2. **Saturation**: Colored LEDs have saturation > 40
3. **Temporal variance**: LED regions change over time (on/off)

### Pattern Classification

| Pattern | Definition |
|---------|------------|
| `blink` | Instant on/off (< 100ms rise/fall time) |
| `solid` | Constant brightness (< 20% variation) |
| `fade_in` | Gradual rise (> 100ms), instant fall |
| `fade_out` | Instant rise, gradual fall (> 100ms) |
| `fade` | Gradual rise AND fall |

### Color Classification

Colors are classified using HSV hue ranges:

| Color | Hue Range (0-180) |
|-------|-------------------|
| Red | 0-10 or 170-180 |
| Orange | 10-25 |
| Yellow | 25-35 |
| Green | 35-85 |
| Cyan | 85-100 |
| Blue | 100-130 |
| Magenta | 130-170 |
| White | Any hue, saturation < 40 |

## Tips for Best Results

### Recording Setup

- 📹 **Frame rate**: 30+ FPS recommended for accurate timing
- 📏 **Distance**: Keep LED clearly visible (not too far)
- 💡 **Lighting**: Avoid direct sunlight on lens
- 📱 **Stability**: Use tripod or stable surface

### Troubleshooting

| Issue | Solution |
|-------|----------|
| LED not detected | Lower `--threshold` value |
| Wrong region detected | Use `--roi` to manually specify |
| Colors misclassified | Check camera white balance |
| Too many false events | Increase `--threshold` |

## Use Cases

- 🚪 Smart door lock LED pattern documentation
- 🔌 IoT device status indicator analysis
- 🎮 Gaming peripheral LED effect capture
- 🏭 Industrial equipment status monitoring
- 📱 Device UI/UX testing automation

## Requirements

- Python 3.8+
- OpenCV 4.0+
- NumPy

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Edwin Isac ([@edwiniac](https://github.com/edwiniac))
