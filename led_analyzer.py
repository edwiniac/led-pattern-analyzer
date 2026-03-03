#!/usr/bin/env python3
"""
LED Pattern Analyzer
====================
Detects LED colors, patterns (blink/fade/solid), and timing from video.
Designed for smart door lock LED indicators but works with any LED in frame.

Author: Edwin Isac
License: MIT
"""

import cv2
import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path


@dataclass
class LEDState:
    """Represents LED state at a single frame."""
    frame_num: int
    timestamp_ms: float
    is_on: bool
    color: str  # 'off', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'white'
    brightness: float  # 0-1 normalized (max in ROI)
    bright_pixel_count: int  # Number of pixels above threshold
    hue: float  # 0-180 (OpenCV HSV) - dominant hue of bright pixels
    saturation: float  # 0-255


@dataclass
class LEDEvent:
    """Represents a detected LED event (on period)."""
    start_ms: float
    end_ms: float
    duration_ms: float
    color: str
    pattern: str  # 'blink', 'fade_in', 'fade_out', 'fade', 'solid'
    avg_brightness: float
    peak_brightness: float
    avg_bright_pixels: float
    rise_time_ms: Optional[float] = None
    fall_time_ms: Optional[float] = None


@dataclass 
class AnalysisResult:
    """Complete analysis result."""
    video_path: str
    fps: float
    total_frames: int
    duration_ms: float
    led_roi: Tuple[int, int, int, int]
    events: List[LEDEvent] = field(default_factory=list)
    color_summary: Dict[str, int] = field(default_factory=dict)
    pattern_summary: Dict[str, int] = field(default_factory=dict)
    color_durations: Dict[str, float] = field(default_factory=dict)
    timeline: List[dict] = field(default_factory=list)


class LEDAnalyzer:
    """
    Analyzes LED patterns in video.
    
    Detection is based on brightness and color, not shape.
    Works with any LED shape: circular, pill, strip, ring, etc.
    """
    
    # Thresholds - tuned for door lock LEDs
    BRIGHT_PIXEL_THRESHOLD = 220  # Pixel value to be considered "bright"
    MIN_BRIGHT_PIXELS = 300  # Minimum bright pixels for LED to be "on"
    BLINK_RISE_TIME_MS = 100  # Faster than this = blink
    MIN_EVENT_DURATION_MS = 30  # Ignore shorter events (noise)
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_ms = (self.total_frames / self.fps) * 1000
        
        self.led_roi = None
        self.states: List[LEDState] = []
        
    def detect_led_roi(self, sample_frames: int = 60) -> Tuple[int, int, int, int]:
        """
        Auto-detect LED region by finding bright, saturated blobs that vary over time.
        
        The detection is shape-agnostic - it finds the LED by looking for:
        1. High brightness (V channel in HSV)
        2. Color saturation (S channel in HSV)  
        3. Temporal variance (LED turns on/off)
        """
        print("🔍 Auto-detecting LED region...")
        
        frame_indices = np.linspace(0, self.total_frames - 1, sample_frames, dtype=int)
        
        # Collect bright pixel locations across frames
        all_bright_coords = []
        
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Look for bright, saturated pixels (LED characteristics)
            led_mask = (v > 220) & (s > 40)
            coords = np.where(led_mask)
            
            if len(coords[0]) > 100:  # Significant LED activity
                all_bright_coords.extend(zip(coords[0], coords[1]))
        
        if not all_bright_coords:
            # Fallback: use center of frame
            self.led_roi = (self.width//4, self.height//4, self.width//2, self.height//2)
            print("⚠️  Warning: No LED detected, using center region")
            return self.led_roi
        
        coords_array = np.array(all_bright_coords)
        y_coords, x_coords = coords_array[:, 0], coords_array[:, 1]
        
        # Find bounding box with some padding
        padding = 30
        x_min = max(0, int(np.percentile(x_coords, 5)) - padding)
        x_max = min(self.width, int(np.percentile(x_coords, 95)) + padding)
        y_min = max(0, int(np.percentile(y_coords, 5)) - padding)
        y_max = min(self.height, int(np.percentile(y_coords, 95)) + padding)
        
        self.led_roi = (x_min, y_min, x_max - x_min, y_max - y_min)
        print(f"✅ Detected LED ROI: x={x_min}, y={y_min}, w={x_max-x_min}, h={y_max-y_min}")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self.led_roi
    
    def set_roi(self, x: int, y: int, w: int, h: int):
        """Manually set LED region of interest."""
        self.led_roi = (x, y, w, h)
    
    def classify_color(self, hue: float, saturation: float) -> str:
        """
        Classify color from HSV values.
        
        OpenCV uses hue range 0-180 (not 0-360).
        """
        if saturation < 40:
            return 'white'
        
        # OpenCV uses hue 0-180
        if hue < 10 or hue > 170:
            return 'red'
        elif hue < 25:
            return 'orange'
        elif hue < 35:
            return 'yellow'
        elif hue < 85:
            return 'green'
        elif hue < 100:
            return 'cyan'
        elif hue < 130:
            return 'blue'
        else:
            return 'magenta'
    
    def analyze_frame(self, frame: np.ndarray, frame_num: int) -> LEDState:
        """Analyze a single frame for LED state."""
        x, y, w, h = self.led_roi
        roi = frame[y:y+h, x:x+w]
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv_roi)
        
        # Count bright pixels
        bright_mask = v_ch > self.BRIGHT_PIXEL_THRESHOLD
        bright_pixel_count = np.sum(bright_mask)
        
        # Max brightness
        max_brightness = np.max(v_ch) / 255.0
        
        # LED is "on" if enough bright pixels
        is_on = bright_pixel_count >= self.MIN_BRIGHT_PIXELS
        
        if is_on and np.any(bright_mask):
            # Get dominant hue and saturation of bright pixels
            hue = np.median(h_ch[bright_mask])
            saturation = np.median(s_ch[bright_mask])
            color = self.classify_color(hue, saturation)
        else:
            hue = 0
            saturation = 0
            color = 'off'
        
        timestamp_ms = (frame_num / self.fps) * 1000
        
        return LEDState(
            frame_num=frame_num,
            timestamp_ms=timestamp_ms,
            is_on=is_on,
            color=color,
            brightness=max_brightness,
            bright_pixel_count=bright_pixel_count,
            hue=hue,
            saturation=saturation
        )
    
    def analyze_video(self, progress_interval: int = 100) -> List[LEDState]:
        """Analyze entire video frame by frame."""
        if self.led_roi is None:
            self.detect_led_roi()
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.states = []
        
        print(f"📊 Analyzing {self.total_frames} frames at {self.fps:.1f} FPS...")
        
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            state = self.analyze_frame(frame, frame_num)
            self.states.append(state)
            
            if frame_num % progress_interval == 0:
                pct = 100 * frame_num / self.total_frames
                print(f"   Frame {frame_num}/{self.total_frames} ({pct:.1f}%)")
            
            frame_num += 1
        
        print(f"✅ Analyzed {len(self.states)} frames")
        return self.states
    
    def detect_events(self) -> List[LEDEvent]:
        """
        Detect LED events from state timeline.
        Groups consecutive 'on' states by color into events.
        """
        if not self.states:
            return []
        
        events = []
        current_event_states = []
        current_color = None
        
        for state in self.states:
            if state.is_on:
                # Check if color changed (new event)
                if current_color is not None and state.color != current_color:
                    # Save current event
                    event = self._create_event(current_event_states)
                    if event and event.duration_ms >= self.MIN_EVENT_DURATION_MS:
                        events.append(event)
                    current_event_states = []
                
                current_event_states.append(state)
                current_color = state.color
            else:
                if current_event_states:
                    event = self._create_event(current_event_states)
                    if event and event.duration_ms >= self.MIN_EVENT_DURATION_MS:
                        events.append(event)
                    current_event_states = []
                    current_color = None
        
        # Handle event at end of video
        if current_event_states:
            event = self._create_event(current_event_states)
            if event and event.duration_ms >= self.MIN_EVENT_DURATION_MS:
                events.append(event)
        
        return events
    
    def _create_event(self, states: List[LEDState]) -> Optional[LEDEvent]:
        """Create an event from consecutive on-states."""
        if not states:
            return None
        
        start_ms = states[0].timestamp_ms
        end_ms = states[-1].timestamp_ms
        frame_duration = 1000 / self.fps
        duration_ms = end_ms - start_ms + frame_duration
        
        # Dominant color (most frequent)
        color_counts = {}
        for s in states:
            color_counts[s.color] = color_counts.get(s.color, 0) + 1
        dominant_color = max(color_counts, key=color_counts.get)
        
        # Brightness stats
        brightnesses = [s.brightness for s in states]
        bright_pixels = [s.bright_pixel_count for s in states]
        avg_brightness = np.mean(brightnesses)
        peak_brightness = np.max(brightnesses)
        avg_bright_pixels = np.mean(bright_pixels)
        
        # Rise and fall times
        rise_time_ms = self._calculate_rise_time(states)
        fall_time_ms = self._calculate_fall_time(states)
        
        # Classify pattern
        pattern = self._classify_pattern(states, rise_time_ms, fall_time_ms)
        
        return LEDEvent(
            start_ms=start_ms,
            end_ms=end_ms,
            duration_ms=duration_ms,
            color=dominant_color,
            pattern=pattern,
            avg_brightness=avg_brightness,
            peak_brightness=peak_brightness,
            avg_bright_pixels=avg_bright_pixels,
            rise_time_ms=rise_time_ms,
            fall_time_ms=fall_time_ms
        )
    
    def _calculate_rise_time(self, states: List[LEDState]) -> Optional[float]:
        """Calculate time from 10% to 90% of peak brightness."""
        if len(states) < 3:
            return None
        
        bright_pixels = [s.bright_pixel_count for s in states]
        peak = max(bright_pixels)
        if peak == 0:
            return None
            
        threshold_10 = peak * 0.1
        threshold_90 = peak * 0.9
        
        time_10, time_90 = None, None
        
        for s in states:
            if time_10 is None and s.bright_pixel_count >= threshold_10:
                time_10 = s.timestamp_ms
            if time_10 is not None and s.bright_pixel_count >= threshold_90:
                time_90 = s.timestamp_ms
                break
        
        if time_10 is not None and time_90 is not None:
            return time_90 - time_10
        return None
    
    def _calculate_fall_time(self, states: List[LEDState]) -> Optional[float]:
        """Calculate time from 90% to 10% of peak brightness (from peak onwards)."""
        if len(states) < 3:
            return None
        
        bright_pixels = [s.bright_pixel_count for s in states]
        peak_idx = np.argmax(bright_pixels)
        peak = bright_pixels[peak_idx]
        if peak == 0:
            return None
        
        threshold_90 = peak * 0.9
        threshold_10 = peak * 0.1
        
        time_90, time_10 = None, None
        
        for s in states[peak_idx:]:
            if time_90 is None and s.bright_pixel_count <= threshold_90:
                time_90 = s.timestamp_ms
            if time_90 is not None and s.bright_pixel_count <= threshold_10:
                time_10 = s.timestamp_ms
                break
        
        if time_90 is not None and time_10 is not None:
            return time_10 - time_90
        return None
    
    def _classify_pattern(self, states: List[LEDState], rise_time: Optional[float], 
                          fall_time: Optional[float]) -> str:
        """
        Classify the LED pattern type.
        
        Patterns:
        - blink: instant on/off (< 100ms rise/fall)
        - solid: constant brightness
        - fade_in: gradual rise, instant fall
        - fade_out: instant rise, gradual fall
        - fade: gradual rise and fall
        """
        bright_pixels = [s.bright_pixel_count for s in states]
        
        if len(bright_pixels) < 3:
            return 'blink'
        
        # Check if brightness is relatively constant (solid)
        bp_std = np.std(bright_pixels)
        bp_mean = np.mean(bright_pixels)
        cv = bp_std / bp_mean if bp_mean > 0 else 0
        
        if cv < 0.2:  # Low variation = solid
            return 'solid'
        
        # Check rise/fall times
        is_fast_rise = rise_time is None or rise_time < self.BLINK_RISE_TIME_MS
        is_fast_fall = fall_time is None or fall_time < self.BLINK_RISE_TIME_MS
        
        if is_fast_rise and is_fast_fall:
            return 'blink'
        elif not is_fast_rise and is_fast_fall:
            return 'fade_in'
        elif is_fast_rise and not is_fast_fall:
            return 'fade_out'
        else:
            return 'fade'
    
    def generate_report(self) -> AnalysisResult:
        """Generate complete analysis report."""
        events = self.detect_events()
        
        # Color summary (count)
        color_summary = {}
        color_durations = {}
        for event in events:
            color_summary[event.color] = color_summary.get(event.color, 0) + 1
            color_durations[event.color] = color_durations.get(event.color, 0) + event.duration_ms
        
        # Pattern summary
        pattern_summary = {}
        for event in events:
            pattern_summary[event.pattern] = pattern_summary.get(event.pattern, 0) + 1
        
        # Create timeline
        timeline = []
        for event in events:
            timeline.append({
                'start_sec': round(event.start_ms / 1000, 3),
                'end_sec': round(event.end_ms / 1000, 3),
                'duration_ms': round(event.duration_ms, 1),
                'color': event.color,
                'pattern': event.pattern,
                'brightness': round(event.avg_brightness, 2)
            })
        
        return AnalysisResult(
            video_path=self.video_path,
            fps=self.fps,
            total_frames=self.total_frames,
            duration_ms=self.duration_ms,
            led_roi=self.led_roi,
            events=events,
            color_summary=color_summary,
            pattern_summary=pattern_summary,
            color_durations=color_durations,
            timeline=timeline
        )
    
    def print_report(self, result: AnalysisResult):
        """Print human-readable report."""
        print("\n" + "=" * 70)
        print("LED PATTERN ANALYSIS REPORT")
        print("=" * 70)
        
        print(f"\n📹 Video: {result.video_path}")
        print(f"   Duration: {result.duration_ms/1000:.2f} seconds")
        print(f"   FPS: {result.fps:.1f}")
        print(f"   Total Frames: {result.total_frames}")
        print(f"   LED ROI: x={result.led_roi[0]}, y={result.led_roi[1]}, "
              f"w={result.led_roi[2]}, h={result.led_roi[3]}")
        
        print(f"\n{'─' * 70}")
        print("📊 SUMMARY")
        print(f"{'─' * 70}")
        print(f"Total LED events: {len(result.events)}")
        
        print(f"\n🎨 Colors detected:")
        for color, count in sorted(result.color_summary.items(), key=lambda x: -x[1]):
            duration = result.color_durations.get(color, 0)
            print(f"   {color:10s}: {count:3d} events, total {duration/1000:.2f}s")
        
        print(f"\n⚡ Patterns detected:")
        for pattern, count in sorted(result.pattern_summary.items(), key=lambda x: -x[1]):
            print(f"   {pattern:10s}: {count:3d} events")
        
        print(f"\n{'─' * 70}")
        print("📅 EVENT TIMELINE")
        print(f"{'─' * 70}")
        
        for i, event in enumerate(result.events, 1):
            rise_str = f"{event.rise_time_ms:.0f}ms" if event.rise_time_ms else "instant"
            fall_str = f"{event.fall_time_ms:.0f}ms" if event.fall_time_ms else "instant"
            
            # Color emoji
            color_emoji = {
                'red': '🔴', 'orange': '🟠', 'yellow': '🟡', 'green': '🟢',
                'cyan': '🔵', 'blue': '🔵', 'magenta': '🟣', 'white': '⚪'
            }.get(event.color, '⬜')
            
            print(f"\n{color_emoji} Event {i}: {event.color.upper()}")
            print(f"   Time: {event.start_ms/1000:.3f}s → {event.end_ms/1000:.3f}s "
                  f"(duration: {event.duration_ms:.0f}ms)")
            print(f"   Pattern: {event.pattern}")
            print(f"   Rise: {rise_str}, Fall: {fall_str}")
        
        # Frequency analysis
        print(f"\n{'─' * 70}")
        print("📈 FREQUENCY ANALYSIS")
        print(f"{'─' * 70}")
        
        by_color = {}
        for event in result.events:
            if event.color not in by_color:
                by_color[event.color] = []
            by_color[event.color].append(event)
        
        for color, events in by_color.items():
            print(f"\n{color.upper()}:")
            print(f"   Count: {len(events)}")
            
            durations = [e.duration_ms for e in events]
            print(f"   Duration: avg={np.mean(durations):.0f}ms, "
                  f"min={np.min(durations):.0f}ms, max={np.max(durations):.0f}ms")
            
            if len(events) >= 2:
                intervals = []
                for i in range(1, len(events)):
                    interval = events[i].start_ms - events[i-1].end_ms
                    if interval > 0:
                        intervals.append(interval)
                
                if intervals:
                    avg_interval = np.mean(intervals)
                    print(f"   Gap between events: avg={avg_interval:.0f}ms")
        
        print("\n" + "=" * 70)
    
    def export_json(self, result: AnalysisResult, output_path: str):
        """Export analysis to JSON file."""
        data = {
            'video_path': result.video_path,
            'fps': result.fps,
            'total_frames': result.total_frames,
            'duration_ms': result.duration_ms,
            'led_roi': list(result.led_roi),
            'summary': {
                'total_events': len(result.events),
                'colors': result.color_summary,
                'patterns': result.pattern_summary,
                'color_durations_ms': result.color_durations
            },
            'timeline': result.timeline,
            'events': [
                {
                    'event_num': i + 1,
                    'start_ms': round(e.start_ms, 1),
                    'end_ms': round(e.end_ms, 1),
                    'duration_ms': round(e.duration_ms, 1),
                    'color': e.color,
                    'pattern': e.pattern,
                    'avg_brightness': round(e.avg_brightness, 3),
                    'peak_brightness': round(e.peak_brightness, 3),
                    'rise_time_ms': round(e.rise_time_ms, 1) if e.rise_time_ms else None,
                    'fall_time_ms': round(e.fall_time_ms, 1) if e.fall_time_ms else None
                }
                for i, e in enumerate(result.events)
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Exported to: {output_path}")
    
    def create_annotated_video(self, result: AnalysisResult, output_path: str):
        """Create video with LED state annotations overlay."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # Color map for drawing
        color_bgr = {
            'off': (128, 128, 128),
            'red': (0, 0, 255),
            'orange': (0, 165, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'cyan': (255, 255, 0),
            'blue': (255, 0, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255)
        }
        
        print(f"🎬 Creating annotated video...")
        
        for i, state in enumerate(self.states):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, state.frame_num)
            ret, frame = self.cap.read()
            if not ret:
                break
            
            x, y, w, h = self.led_roi
            color = color_bgr.get(state.color, (128, 128, 128))
            thickness = 3 if state.is_on else 1
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            # Status text
            status = f"{state.color.upper()}" if state.is_on else "OFF"
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, color, 2)
            
            # Timestamp
            time_text = f"T: {state.timestamp_ms/1000:.2f}s"
            cv2.putText(frame, time_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)
            
            # Bright pixel count
            px_text = f"Bright px: {state.bright_pixel_count}"
            cv2.putText(frame, px_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            
            out.write(frame)
            
            if i % 100 == 0:
                print(f"   Frame {i}/{len(self.states)}")
        
        out.release()
        print(f"✅ Annotated video saved: {output_path}")
    
    def close(self):
        """Release video capture."""
        self.cap.release()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze LED patterns in video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect LED and analyze
  python led_analyzer.py video.mp4
  
  # Specify output file
  python led_analyzer.py video.mp4 -o results.json
  
  # Manual ROI (if auto-detection fails)
  python led_analyzer.py video.mp4 --roi 400,800,300,400
  
  # Adjust sensitivity (lower = more sensitive)
  python led_analyzer.py video.mp4 --threshold 200
  
  # Create annotated video for verification
  python led_analyzer.py video.mp4 --annotate
        """
    )
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Output JSON path (default: same as video with .json)')
    parser.add_argument('--annotate', '-a', help='Create annotated video', action='store_true')
    parser.add_argument('--roi', help='Manual ROI as x,y,w,h (e.g., 400,800,300,400)')
    parser.add_argument('--threshold', '-t', type=int, default=300,
                        help='Min bright pixels for LED on detection (default: 300)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = LEDAnalyzer(args.video)
    analyzer.MIN_BRIGHT_PIXELS = args.threshold
    
    # Set ROI if provided
    if args.roi:
        x, y, w, h = map(int, args.roi.split(','))
        analyzer.set_roi(x, y, w, h)
    
    # Run analysis
    analyzer.analyze_video()
    result = analyzer.generate_report()
    analyzer.print_report(result)
    
    # Export JSON
    if args.output:
        analyzer.export_json(result, args.output)
    else:
        video_path = Path(args.video)
        json_path = video_path.with_suffix('.json')
        analyzer.export_json(result, str(json_path))
    
    # Create annotated video if requested
    if args.annotate:
        video_path = Path(args.video)
        annotated_path = video_path.parent / f"{video_path.stem}_annotated.mp4"
        analyzer.create_annotated_video(result, str(annotated_path))
    
    analyzer.close()


if __name__ == '__main__':
    main()
