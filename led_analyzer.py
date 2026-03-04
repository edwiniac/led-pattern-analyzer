#!/usr/bin/env python3
"""
LED Pattern Analyzer - Detects LED colors and patterns from video.
"""

import cv2
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class LEDEvent:
    start_sec: float
    end_sec: float
    duration_ms: float
    color: str
    pattern: str


class LEDAnalyzer:
    BRIGHTNESS_THRESHOLD = 200
    SATURATION_THRESHOLD = 60
    MIN_PIXELS = 100
    MIN_EVENT_MS = 25
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.roi = None
    
    def detect_led_roi(self) -> Tuple[int, int, int, int]:
        """
        Find LED by looking for the region with most brightness CHANGES.
        LED turns on/off, ambient lights stay constant.
        """
        print("Detecting LED region (looking for brightness changes)...")
        
        sample_indices = np.linspace(0, self.total_frames - 1, 60, dtype=int)
        grid_size = 50
        
        # Track brightness per grid cell across frames
        grid_brightness = {}  # {(y,x): [brightness values]}
        
        for idx in sample_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            s = hsv[:, :, 1]
            
            # Check each grid cell
            for gy in range(0, self.height, grid_size):
                for gx in range(0, self.width, grid_size):
                    cell = v[gy:gy+grid_size, gx:gx+grid_size]
                    sat_cell = s[gy:gy+grid_size, gx:gx+grid_size]
                    
                    # Only consider cells that sometimes have saturated bright pixels
                    bright_sat = np.sum((cell > 200) & (sat_cell > 50))
                    
                    key = (gy // grid_size, gx // grid_size)
                    if key not in grid_brightness:
                        grid_brightness[key] = []
                    grid_brightness[key].append(bright_sat)
        
        # Find cells with HIGH VARIANCE (LED turns on/off)
        cell_variance = {}
        for key, values in grid_brightness.items():
            if max(values) > 10:  # Must have some bright frames
                cell_variance[key] = np.std(values)
        
        if not cell_variance:
            print("Warning: No LED detected, using center region")
            self.roi = (self.width//4, self.height//4, self.width//2, self.height//2)
            return self.roi
        
        # Get top cells by variance
        sorted_cells = sorted(cell_variance.keys(), key=lambda k: cell_variance[k], reverse=True)
        top_cells = sorted_cells[:5]  # Top 5 most variable cells
        
        # Build ROI
        y_cells = [c[0] for c in top_cells]
        x_cells = [c[1] for c in top_cells]
        
        padding = 1
        y_min = max(0, (min(y_cells) - padding) * grid_size)
        y_max = min(self.height, (max(y_cells) + padding + 1) * grid_size)
        x_min = max(0, (min(x_cells) - padding) * grid_size)
        x_max = min(self.width, (max(x_cells) + padding + 1) * grid_size)
        
        self.roi = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        print(f"LED ROI: x={x_min}, y={y_min}, w={x_max-x_min}, h={y_max-y_min}")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self.roi
    
    def set_roi(self, x: int, y: int, w: int, h: int):
        self.roi = (x, y, w, h)
    
    def _classify_color_rgb(self, r: float, g: float, b: float) -> str:
        # White = RGB balanced AND very bright (min > 210)
        rgb_range = max(r, g, b) - min(r, g, b)
        if rgb_range < 35 and min(r, g, b) > 210:
            return 'white'
        # Colored - which channel dominates?
        if b > r and b > g:
            return 'blue'
        elif g > r and g > b:
            return 'green'
        else:
            return 'red'
    
    def _get_led_state(self, frame: np.ndarray) -> Tuple[bool, Optional[str], int]:
        x, y, w, h = self.roi
        roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_ch = hsv[:, :, 2]
        
        # Bright pixels (LED on)
        bright_mask = v_ch > self.BRIGHTNESS_THRESHOLD
        pixel_count = np.sum(bright_mask)
        
        if pixel_count < self.MIN_PIXELS:
            return False, None, pixel_count
        
        # Get median RGB of bright pixels
        b = np.median(roi[:, :, 0][bright_mask])
        g = np.median(roi[:, :, 1][bright_mask])
        r = np.median(roi[:, :, 2][bright_mask])
        
        return True, self._classify_color_rgb(r, g, b), pixel_count
    
    def _classify_pattern(self, pixels: List[int], duration_ms: float) -> str:
        if len(pixels) < 2:
            return 'blink'
        cv = np.std(pixels) / np.mean(pixels) if np.mean(pixels) > 0 else 0
        if cv > 0.3:
            return 'fade'
        elif duration_ms < 150:
            return 'blink'
        return 'solid'
    
    def _merge_events(self, events: List[LEDEvent], gap_ms: float = 150) -> List[LEDEvent]:
        """Merge events of same color that are close together, absorb short blinks."""
        if len(events) < 2:
            return events
        
        merged = []
        i = 0
        
        while i < len(events):
            current = events[i]
            
            # Look ahead for events to merge
            j = i + 1
            while j < len(events):
                next_evt = events[j]
                gap = (next_evt.start_sec - current.end_sec) * 1000
                
                # Merge if: same color and small gap, OR next is short blink (< 100ms)
                if gap < gap_ms:
                    if next_evt.color == current.color:
                        # Same color - merge
                        current = LEDEvent(
                            start_sec=current.start_sec,
                            end_sec=next_evt.end_sec,
                            duration_ms=(next_evt.end_sec - current.start_sec) * 1000,
                            color=current.color,
                            pattern='fade' if current.pattern == 'fade' or next_evt.pattern == 'fade' else current.pattern
                        )
                        j += 1
                    elif next_evt.duration_ms < 100:
                        # Short blink of different color - absorb it
                        # Check if there's another event of current color after
                        if j + 1 < len(events) and events[j + 1].color == current.color:
                            k = j + 1
                            if (events[k].start_sec - current.end_sec) * 1000 < gap_ms * 2:
                                current = LEDEvent(
                                    start_sec=current.start_sec,
                                    end_sec=events[k].end_sec,
                                    duration_ms=(events[k].end_sec - current.start_sec) * 1000,
                                    color=current.color,
                                    pattern='fade'
                                )
                                j = k + 1
                                continue
                        break
                    else:
                        break
                else:
                    break
            
            merged.append(current)
            i = j if j > i + 1 else i + 1
        
        return merged
    
    def analyze(self) -> List[LEDEvent]:
        if self.roi is None:
            self.detect_led_roi()
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        events = []
        current_color = None
        event_start = None
        event_pixels = []
        
        for frame_num in range(self.total_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            is_on, color, pixels = self._get_led_state(frame)
            
            if is_on:
                if current_color is None:
                    current_color = color
                    event_start = frame_num
                    event_pixels = [pixels]
                elif color == current_color:
                    event_pixels.append(pixels)
                else:
                    events.append(self._make_event(event_start, frame_num-1, current_color, event_pixels))
                    current_color = color
                    event_start = frame_num
                    event_pixels = [pixels]
            else:
                if current_color:
                    events.append(self._make_event(event_start, frame_num-1, current_color, event_pixels))
                    current_color = None
                    event_pixels = []
        
        if current_color:
            events.append(self._make_event(event_start, self.total_frames-1, current_color, event_pixels))
        
        events = [e for e in events if e and e.duration_ms >= self.MIN_EVENT_MS]
        return self._merge_events(events)
    
    def _make_event(self, start: int, end: int, color: str, pixels: List[int]) -> Optional[LEDEvent]:
        start_sec = start / self.fps
        end_sec = end / self.fps
        duration_ms = (end_sec - start_sec) * 1000 + (1000 / self.fps)
        return LEDEvent(
            start_sec=round(start_sec, 3),
            end_sec=round(end_sec, 3),
            duration_ms=round(duration_ms, 1),
            color=color,
            pattern=self._classify_pattern(pixels, duration_ms)
        )
    
    def close(self):
        self.cap.release()


def analyze_video(video_path: str, output_path: str = None, roi: str = None) -> dict:
    analyzer = LEDAnalyzer(video_path)
    
    if roi:
        x, y, w, h = map(int, roi.split(','))
        analyzer.set_roi(x, y, w, h)
    
    events = analyzer.analyze()
    
    results = {
        'video': video_path,
        'duration_sec': round(analyzer.total_frames / analyzer.fps, 2),
        'fps': round(analyzer.fps, 1),
        'roi': [int(x) for x in analyzer.roi],
        'total_events': len(events),
        'events': [{'start_sec': e.start_sec, 'end_sec': e.end_sec, 'duration_ms': e.duration_ms, 
                    'color': e.color, 'pattern': e.pattern} for e in events]
    }
    
    # Summary
    by_color = {}
    for e in events:
        if e.color not in by_color:
            by_color[e.color] = {'count': 0, 'total_ms': 0}
        by_color[e.color]['count'] += 1
        by_color[e.color]['total_ms'] += e.duration_ms
    results['by_color'] = by_color
    
    # Pattern sequence
    results['pattern_sequence'] = [
        f"{e.color}-{e.pattern}({e.duration_ms:.0f}ms)" for e in events
    ]
    results['pattern_compact'] = " → ".join([
        f"{e.color[0].upper()}-{e.pattern[0]}" for e in events
    ])
    
    # Print
    print(f"\n{'='*50}\nLED PATTERN ANALYSIS\n{'='*50}")
    print(f"Video: {video_path}")
    print(f"Duration: {results['duration_sec']}s | FPS: {results['fps']}")
    print(f"ROI: {analyzer.roi}")
    print(f"\nTotal events: {len(events)}\n")
    
    for color, stats in by_color.items():
        print(f"  {color.upper()}: {stats['count']} events, {stats['total_ms']/1000:.2f}s")
    
    print(f"\nEvents:")
    for i, e in enumerate(events, 1):
        print(f"  {i:2d}. [{e.start_sec:6.2f}s - {e.end_sec:6.2f}s] {e.color.upper():6s} {e.pattern:6s} ({e.duration_ms:.0f}ms)")
    
    # Generate pattern sequence string
    print(f"\n{'='*50}")
    print("PATTERN SEQUENCE")
    print('='*50)
    
    # Compact format: color-pattern (duration)
    pattern_parts = []
    for e in events:
        duration_str = f"{e.duration_ms:.0f}ms" if e.duration_ms < 1000 else f"{e.duration_ms/1000:.1f}s"
        pattern_parts.append(f"{e.color}-{e.pattern}({duration_str})")
    
    # Print in readable lines
    line = ""
    for i, part in enumerate(pattern_parts):
        if len(line) + len(part) > 70:
            print(line)
            line = part
        else:
            line = line + " → " + part if line else part
    if line:
        print(line)
    
    # Also print ultra-compact version
    print(f"\nCompact: ", end="")
    compact = " → ".join([f"{e.color[0].upper()}-{e.pattern[0]}" for e in events])
    print(compact)
    
    print('='*50)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {output_path}")
    
    analyzer.close()
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze LED patterns in video')
    parser.add_argument('video', help='Video file')
    parser.add_argument('-o', '--output', help='Output JSON')
    parser.add_argument('--roi', help='Manual ROI: x,y,w,h')
    args = parser.parse_args()
    
    output = args.output or str(Path(args.video).with_suffix('.json'))
    analyze_video(args.video, output, args.roi)


if __name__ == '__main__':
    main()
