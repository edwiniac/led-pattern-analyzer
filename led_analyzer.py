#!/usr/bin/env python3
"""
LED Pattern Analyzer
====================
Detects LED colors and patterns from video.
Outputs clean, focused LED event data.

Author: Edwin Isac
License: MIT
"""

import cv2
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class LEDEvent:
    """A single LED event."""
    start_sec: float
    end_sec: float
    duration_ms: float
    color: str
    pattern: str  # 'blink', 'solid', 'fade'


class LEDAnalyzer:
    """Analyzes LED patterns in video."""
    
    # Detection thresholds
    BRIGHTNESS_THRESHOLD = 200  # V channel
    SATURATION_THRESHOLD = 60   # S channel - filters out white/ambient
    MIN_PIXELS = 200            # Minimum LED pixels to be "on"
    MIN_EVENT_MS = 25           # Ignore events shorter than this
    BLINK_THRESHOLD_MS = 80     # Faster than this = blink
    
    def __init__(self, video_path: str, colors: List[str] = None):
        """
        Initialize analyzer.
        
        Args:
            video_path: Path to video file
            colors: Expected colors to detect (e.g., ['red', 'green']). 
                   If None, detects all colors.
        """
        self.video_path = video_path
        self.expected_colors = colors
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.total_frames / self.fps
    
    def _classify_color(self, hue: float) -> str:
        """
        Classify color from HSV hue (0-180 scale).
        
        Simplified to 3 primary LED colors:
        - RED: hue 0-35 or 150-180 (includes orange tones)
        - GREEN: hue 35-85
        - BLUE: hue 85-150 (includes cyan)
        """
        if hue < 35 or hue > 150:
            return 'red'
        elif hue < 85:
            return 'green'
        else:
            return 'blue'
    
    def _get_led_state(self, frame: np.ndarray) -> Tuple[bool, Optional[str], int]:
        """
        Get LED state from a frame.
        
        Returns: (is_on, color, bright_pixel_count)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # LED mask: bright AND saturated (filters ambient light)
        led_mask = (v > self.BRIGHTNESS_THRESHOLD) & (s > self.SATURATION_THRESHOLD)
        pixel_count = np.sum(led_mask)
        
        if pixel_count < self.MIN_PIXELS:
            return False, None, pixel_count
        
        # Get dominant color
        hue = np.median(h[led_mask])
        color = self._classify_color(hue)
        
        # Filter to expected colors if specified
        if self.expected_colors and color not in self.expected_colors:
            # Try to map to nearest expected color
            if 'red' in self.expected_colors and (hue < 30 or hue > 150):
                color = 'red'
            elif 'green' in self.expected_colors and 35 <= hue <= 85:
                color = 'green'
            elif 'blue' in self.expected_colors and 85 < hue < 130:
                color = 'blue'
            else:
                return False, None, pixel_count
        
        return True, color, pixel_count
    
    def _classify_pattern(self, pixel_counts: List[int], duration_ms: float) -> str:
        """Classify pattern based on brightness profile."""
        if len(pixel_counts) < 2:
            return 'blink'
        
        # Check variance
        std = np.std(pixel_counts)
        mean = np.mean(pixel_counts)
        cv = std / mean if mean > 0 else 0
        
        if cv > 0.3:  # High variance = fade
            return 'fade'
        elif duration_ms < 150:  # Short = blink
            return 'blink'
        else:
            return 'solid'
    
    def analyze(self) -> List[LEDEvent]:
        """
        Analyze video and return LED events.
        
        Returns: List of LEDEvent objects
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        events = []
        current_color = None
        event_start_frame = None
        event_pixels = []
        
        print(f"Analyzing {self.total_frames} frames...")
        
        for frame_num in range(self.total_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            is_on, color, pixels = self._get_led_state(frame)
            
            if is_on:
                if current_color is None:
                    # Start new event
                    current_color = color
                    event_start_frame = frame_num
                    event_pixels = [pixels]
                elif color == current_color:
                    # Continue event
                    event_pixels.append(pixels)
                else:
                    # Color changed - save current, start new
                    event = self._create_event(
                        event_start_frame, frame_num - 1, 
                        current_color, event_pixels
                    )
                    if event:
                        events.append(event)
                    
                    current_color = color
                    event_start_frame = frame_num
                    event_pixels = [pixels]
            else:
                if current_color is not None:
                    # End event
                    event = self._create_event(
                        event_start_frame, frame_num - 1,
                        current_color, event_pixels
                    )
                    if event:
                        events.append(event)
                    
                    current_color = None
                    event_pixels = []
        
        # Handle event at end
        if current_color is not None:
            event = self._create_event(
                event_start_frame, self.total_frames - 1,
                current_color, event_pixels
            )
            if event:
                events.append(event)
        
        print(f"Found {len(events)} LED events")
        return events
    
    def _create_event(self, start_frame: int, end_frame: int, 
                      color: str, pixels: List[int]) -> Optional[LEDEvent]:
        """Create an event from frame range."""
        start_sec = start_frame / self.fps
        end_sec = end_frame / self.fps
        duration_ms = (end_sec - start_sec) * 1000 + (1000 / self.fps)
        
        if duration_ms < self.MIN_EVENT_MS:
            return None
        
        pattern = self._classify_pattern(pixels, duration_ms)
        
        return LEDEvent(
            start_sec=round(start_sec, 3),
            end_sec=round(end_sec, 3),
            duration_ms=round(duration_ms, 1),
            color=color,
            pattern=pattern
        )
    
    def close(self):
        self.cap.release()


def analyze_video(video_path: str, colors: List[str] = None, 
                  output_path: str = None) -> dict:
    """
    Main analysis function.
    
    Args:
        video_path: Path to video
        colors: Expected colors (e.g., ['red', 'green'])
        output_path: Optional JSON output path
    
    Returns:
        Analysis results dict
    """
    analyzer = LEDAnalyzer(video_path, colors=colors)
    events = analyzer.analyze()
    
    # Build results
    results = {
        'video': video_path,
        'duration_sec': round(analyzer.duration_sec, 2),
        'fps': round(analyzer.fps, 1),
        'total_events': len(events),
        'events': [
            {
                'start_sec': e.start_sec,
                'end_sec': e.end_sec,
                'duration_ms': e.duration_ms,
                'color': e.color,
                'pattern': e.pattern
            }
            for e in events
        ]
    }
    
    # Summary by color
    color_stats = {}
    for e in events:
        if e.color not in color_stats:
            color_stats[e.color] = {'count': 0, 'total_ms': 0}
        color_stats[e.color]['count'] += 1
        color_stats[e.color]['total_ms'] += e.duration_ms
    results['by_color'] = color_stats
    
    # Print summary
    print("\n" + "=" * 50)
    print("LED PATTERN ANALYSIS")
    print("=" * 50)
    print(f"\nVideo: {video_path}")
    print(f"Duration: {analyzer.duration_sec:.2f}s | FPS: {analyzer.fps:.1f}")
    print(f"\nTotal events: {len(events)}")
    
    print("\nBy color:")
    for color, stats in color_stats.items():
        print(f"  {color.upper()}: {stats['count']} events, {stats['total_ms']/1000:.2f}s total")
    
    print("\nEvents:")
    for i, e in enumerate(events, 1):
        print(f"  {i:2d}. [{e.start_sec:6.2f}s - {e.end_sec:6.2f}s] "
              f"{e.color.upper():6s} {e.pattern:6s} ({e.duration_ms:.0f}ms)")
    
    print("=" * 50)
    
    # Save JSON
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {output_path}")
    
    analyzer.close()
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze LED patterns in video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python led_analyzer.py video.mp4
  python led_analyzer.py video.mp4 --colors red green
  python led_analyzer.py video.mp4 -o results.json
        """
    )
    parser.add_argument('video', help='Video file path')
    parser.add_argument('-o', '--output', help='Output JSON path')
    parser.add_argument('--colors', nargs='+', 
                        help='Expected colors (e.g., red green)')
    
    args = parser.parse_args()
    
    output = args.output
    if not output:
        output = str(Path(args.video).with_suffix('.json'))
    
    analyze_video(args.video, colors=args.colors, output_path=output)


if __name__ == '__main__':
    main()
