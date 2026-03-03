# LED Pattern Analyzer 💡

Detect LED colors and patterns from video. Designed for smart lock LED indicators.

## Features

- 🎨 **3 Colors**: Red, Green, Blue (robust to lighting variations)
- ⚡ **3 Patterns**: Blink, Solid, Fade
- ⏱️ **Timing**: Start, end, duration for each event
- 📊 **JSON Output**: Clean structured data

## Installation

```bash
git clone https://github.com/edwiniac/led-pattern-analyzer.git
cd led-pattern-analyzer
pip install -r requirements.txt
```

## Usage

```bash
# Analyze video
python led_analyzer.py video.mp4

# Specify output file
python led_analyzer.py video.mp4 -o results.json

# Filter to specific colors only
python led_analyzer.py video.mp4 --colors red green
```

## Output

```
==================================================
LED PATTERN ANALYSIS
==================================================

Video: sample.mp4
Duration: 30.05s | FPS: 30.0

Total events: 33

By color:
  RED: 17 events, 12.43s total
  BLUE: 10 events, 12.03s total
  GREEN: 6 events, 1.97s total

Events:
   1. [  0.00s -   0.13s] RED    solid  (167ms)
   2. [  3.20s -   3.46s] GREEN  solid  (300ms)
   3. [  3.50s -   3.76s] RED    solid  (300ms)
   ...
==================================================
```

### JSON Structure

```json
{
  "video": "sample.mp4",
  "duration_sec": 30.05,
  "fps": 30.0,
  "total_events": 33,
  "by_color": {
    "red": {"count": 17, "total_ms": 12430},
    "green": {"count": 6, "total_ms": 1970},
    "blue": {"count": 10, "total_ms": 12030}
  },
  "events": [
    {
      "start_sec": 0.0,
      "end_sec": 0.13,
      "duration_ms": 167,
      "color": "red",
      "pattern": "solid"
    }
  ]
}
```

## Color Detection

Colors are grouped to handle camera/lighting variance:

| LED Color | Hue Range | Includes |
|-----------|-----------|----------|
| **RED** | 0-35° or 150-180° | Orange tones |
| **GREEN** | 35-85° | Yellow-green |
| **BLUE** | 85-150° | Cyan tones |

## Pattern Detection

| Pattern | Meaning |
|---------|---------|
| `blink` | Short duration (< 150ms) |
| `solid` | Constant brightness |
| `fade` | Gradual brightness change |

## Tips

- Works with any LED shape (circular, strip, ring, etc.)
- 30+ FPS recommended for accurate timing
- Stable camera = better results
- Use `--colors` to filter expected colors

## License

MIT
