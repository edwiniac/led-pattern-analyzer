# LED Pattern Analyzer 💡

Detect LED colors and patterns from video. Built for smart lock testing.

## Installation

```bash
git clone https://github.com/edwiniac/led-pattern-analyzer.git
cd led-pattern-analyzer
pip install -r requirements.txt
```

## Usage

```bash
# With ROI (recommended for accuracy)
python led_analyzer.py video.mp4 --roi x,y,width,height

# Auto-detect ROI (may include unwanted areas)
python led_analyzer.py video.mp4
```

### Finding the ROI

1. Open your video and find a frame where the LED is ON
2. Note the LED position: `x,y` = top-left corner
3. Note the size: `width,height` covering the LED area
4. Use: `--roi x,y,width,height`

Example: `--roi 400,800,200,400`

## Output

```
Total events: 21

  BLUE:  6 events, 11.06s
  GREEN: 6 events, 2.00s
  RED:   9 events, 2.60s

Events:
   1. [  1.47s -   3.03s] BLUE   fade   (1599ms)
   2. [  3.20s -   3.46s] GREEN  solid  (300ms)
   3. [  3.50s -   3.76s] RED    solid  (300ms)
   ...
```

### JSON Output

```json
{
  "video": "sample.mp4",
  "duration_sec": 30.05,
  "total_events": 21,
  "by_color": {
    "blue": {"count": 6, "total_ms": 11060},
    "green": {"count": 6, "total_ms": 2000},
    "red": {"count": 9, "total_ms": 2600}
  },
  "events": [
    {"start_sec": 1.47, "end_sec": 3.03, "duration_ms": 1599, "color": "blue", "pattern": "fade"},
    {"start_sec": 3.20, "end_sec": 3.46, "duration_ms": 300, "color": "green", "pattern": "solid"}
  ]
}
```

## Colors & Patterns

**Colors detected:** RED, GREEN, BLUE (adapts to lighting variations)

**Patterns:**
- `blink` — short flash (< 150ms)
- `solid` — constant brightness
- `fade` — gradual change

## Tips

- Use manual `--roi` for best accuracy
- Keep camera stable during recording
- 30+ FPS recommended
- Works with any LED shape

## License

MIT
