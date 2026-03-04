#!/usr/bin/env python3
"""
LED Pattern Analyzer v3
Detects LED colors, patterns (blink/fade/solid), and timing from video.
Enhanced with adaptive lighting, noise filtering, and pattern grouping.
"""

import cv2
import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from collections import Counter


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
    ambient_brightness: float = 0.0  # Background brightness level


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
    confidence: float = 1.0  # Detection confidence
    rise_time_ms: Optional[float] = None
    fall_time_ms: Optional[float] = None


@dataclass
class PatternSequence:
    """Represents a grouped sequence of events (e.g., alternating colors)."""
    events: List[LEDEvent]
    sequence_type: str  # 'alternating', 'repeating', 'single'
    colors: List[str]
    repeat_count: int
    total_duration_ms: float
    description: str


@dataclass 
class AnalysisResult:
    """Complete analysis result."""
    video_path: str
    fps: float
    total_frames: int
    duration_ms: float
    led_roi: Tuple[int, int, int, int]
    lighting_profile: Dict[str, float]
    events: List[LEDEvent] = field(default_factory=list)
    sequences: List[PatternSequence] = field(default_factory=list)
    color_summary: Dict[str, int] = field(default_factory=dict)
    pattern_summary: Dict[str, int] = field(default_factory=dict)
    color_durations: Dict[str, float] = field(default_factory=dict)
    timeline: List[dict] = field(default_factory=list)
    formatted_timeline: str = ""


class LEDAnalyzer:
    """Analyzes LED patterns in video with adaptive lighting support."""
    
    # Base thresholds (will be adjusted adaptively)
    BASE_BRIGHT_THRESHOLD = 200  # Base pixel value for "bright"
    BASE_MIN_BRIGHT_PIXELS = 200  # Base minimum bright pixels for LED "on"
    BLINK_RISE_TIME_MS = 100  # Faster than this = blink
    MIN_EVENT_DURATION_MS = 30  # Ignore shorter events (noise)
    
    # Temporal smoothing window (frames) - keep small to preserve transitions
    SMOOTHING_WINDOW = 2
    
    # Color classification tuned for LED colors
    # Orange LEDs typically have hue 10-25 (amber/orange glow)
    # Red LEDs are more pure red (hue 0-10 or 170-180)
    COLOR_RANGES = {
        'red': [(0, 10), (170, 180)],  # Pure red (narrow)
        'orange': [(10, 30)],  # Orange/amber LEDs (hue ~14-25)
        'yellow': [(30, 45)],
        'green': [(45, 100)],  # Green LEDs (includes cyan-ish greens)
        'blue': [(100, 135)],
        'magenta': [(135, 170)],
    }
    
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
        
        # Adaptive thresholds (calibrated per video)
        self.bright_threshold = self.BASE_BRIGHT_THRESHOLD
        self.min_bright_pixels = self.BASE_MIN_BRIGHT_PIXELS
        self.ambient_baseline = 0.0
        self.lighting_profile = {}
        
    def calibrate_lighting(self, sample_frames: int = 30) -> Dict[str, float]:
        """
        Calibrate thresholds based on video's lighting conditions.
        Samples frames to understand ambient light levels.
        """
        print("🔆 Calibrating for lighting conditions...")
        
        frame_indices = np.linspace(0, self.total_frames - 1, sample_frames, dtype=int)
        
        brightness_samples = []
        dark_frame_count = 0
        bright_frame_count = 0
        
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            brightness_samples.append(mean_brightness)
            
            if mean_brightness < 50:
                dark_frame_count += 1
            elif mean_brightness > 150:
                bright_frame_count += 1
        
        if not brightness_samples:
            return self._default_lighting_profile()
        
        avg_brightness = np.mean(brightness_samples)
        min_brightness = np.min(brightness_samples)
        max_brightness = np.max(brightness_samples)
        std_brightness = np.std(brightness_samples)
        
        # Determine lighting condition
        if dark_frame_count > len(brightness_samples) * 0.5:
            condition = "dark"
            # Lower thresholds for dark conditions
            self.bright_threshold = max(150, self.BASE_BRIGHT_THRESHOLD - 50)
            self.min_bright_pixels = max(100, self.BASE_MIN_BRIGHT_PIXELS - 100)
        elif bright_frame_count > len(brightness_samples) * 0.5:
            condition = "bright"
            # Higher thresholds to avoid false positives from ambient light
            self.bright_threshold = min(240, self.BASE_BRIGHT_THRESHOLD + 20)
            self.min_bright_pixels = self.BASE_MIN_BRIGHT_PIXELS + 100
        elif std_brightness > 40:
            condition = "variable"
            # Use adaptive per-frame thresholding
            self.bright_threshold = self.BASE_BRIGHT_THRESHOLD
            self.min_bright_pixels = self.BASE_MIN_BRIGHT_PIXELS
        else:
            condition = "normal"
            self.bright_threshold = self.BASE_BRIGHT_THRESHOLD
            self.min_bright_pixels = self.BASE_MIN_BRIGHT_PIXELS
        
        self.ambient_baseline = min_brightness
        
        self.lighting_profile = {
            'condition': condition,
            'avg_brightness': float(avg_brightness),
            'min_brightness': float(min_brightness),
            'max_brightness': float(max_brightness),
            'std_brightness': float(std_brightness),
            'bright_threshold': self.bright_threshold,
            'min_bright_pixels': self.min_bright_pixels,
            'dark_frame_ratio': dark_frame_count / len(brightness_samples),
            'bright_frame_ratio': bright_frame_count / len(brightness_samples),
        }
        
        print(f"   Condition: {condition}")
        print(f"   Avg brightness: {avg_brightness:.1f}")
        print(f"   Adjusted threshold: {self.bright_threshold}")
        print(f"   Min bright pixels: {self.min_bright_pixels}")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self.lighting_profile
    
    def _default_lighting_profile(self) -> Dict[str, float]:
        return {
            'condition': 'unknown',
            'avg_brightness': 100.0,
            'min_brightness': 0.0,
            'max_brightness': 255.0,
            'std_brightness': 50.0,
            'bright_threshold': self.BASE_BRIGHT_THRESHOLD,
            'min_bright_pixels': self.BASE_MIN_BRIGHT_PIXELS,
        }
        
    def detect_led_roi(self, sample_frames: int = 60) -> Tuple[int, int, int, int]:
        """
        Auto-detect LED region by finding bright, saturated blobs that vary over time.
        Enhanced with morphological operations for noise reduction.
        """
        print("🎯 Auto-detecting LED region...")
        
        frame_indices = np.linspace(0, self.total_frames - 1, sample_frames, dtype=int)
        
        all_bright_coords = []
        
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # LED detection mask: bright AND saturated (excludes white cards/hands)
            # High saturation threshold (100) filters out low-saturation objects
            led_mask = (v > self.bright_threshold) & (s > 100)
            
            # Morphological cleanup to remove noise
            kernel = np.ones((3, 3), np.uint8)
            led_mask = cv2.morphologyEx(led_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            led_mask = cv2.morphologyEx(led_mask, cv2.MORPH_CLOSE, kernel)
            
            coords = np.where(led_mask > 0)
            
            if len(coords[0]) > 50:
                all_bright_coords.extend(zip(coords[0], coords[1]))
        
        if not all_bright_coords:
            self.led_roi = (self.width//4, self.height//4, self.width//2, self.height//2)
            print(f"   Warning: No LED detected, using center region")
            return self.led_roi
        
        coords_array = np.array(all_bright_coords)
        y_coords, x_coords = coords_array[:, 0], coords_array[:, 1]
        
        padding = 40
        x_min = max(0, int(np.percentile(x_coords, 2)) - padding)
        x_max = min(self.width, int(np.percentile(x_coords, 98)) + padding)
        y_min = max(0, int(np.percentile(y_coords, 2)) - padding)
        y_max = min(self.height, int(np.percentile(y_coords, 98)) + padding)
        
        self.led_roi = (x_min, y_min, x_max - x_min, y_max - y_min)
        print(f"   ROI: x={x_min}, y={y_min}, w={x_max-x_min}, h={y_max-y_min}")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self.led_roi
    
    def set_roi(self, x: int, y: int, w: int, h: int):
        """Manually set LED region of interest."""
        self.led_roi = (x, y, w, h)
    
    def classify_color(self, hue: float, saturation: float, brightness: float) -> str:
        """
        Classify color from HSV values with improved handling of edge cases.
        Prioritizes detecting actual colors over white.
        Green LEDs often appear desaturated and cyan-shifted - use wider hue range and lower sat threshold.
        """
        # Check for green first with lower saturation threshold (LEDs often look pale/cyan-green)
        # Extended range 50-105 to catch cyan-ish greens
        if 50 <= hue < 105 and saturation >= 12:
            return 'green'
        
        # Check for blue (distinct from green-cyan)
        if 105 <= hue < 135 and saturation >= 15:
            return 'blue'
        
        # Only classify as white if very low saturation
        if saturation < 15:
            return 'white'
        
        # Standard hue-based classification - prioritize color detection
        for color, ranges in self.COLOR_RANGES.items():
            for low, high in ranges:
                if low <= hue < high:
                    # For low saturation (15-40), skip hard-to-distinguish colors
                    if saturation < 40 and color == 'magenta':
                        continue
                    return color
        
        # If saturation is moderate but no color matched, it's likely white
        if saturation < 40:
            return 'white'
        
        return 'white'  # Default fallback
    
    def analyze_frame(self, frame: np.ndarray, frame_num: int) -> LEDState:
        """Analyze a single frame for LED state with adaptive thresholding."""
        x, y, w, h = self.led_roi
        roi = frame[y:y+h, x:x+w]
        
        # Calculate ambient brightness for this frame
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ambient = np.percentile(gray_roi, 20)  # Lower percentile = background level
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv_roi)
        
        # Adaptive threshold based on ambient light
        adaptive_threshold = max(
            self.bright_threshold,
            ambient + 40  # At least 40 above ambient
        )
        
        # Count bright AND saturated pixels (filters out white cards/reflections)
        # MIN_LED_SATURATION = 60 excludes low-saturation objects like white cards
        MIN_LED_SATURATION = 60
        bright_mask = v_ch > adaptive_threshold
        saturated_mask = s_ch > MIN_LED_SATURATION
        led_mask = bright_mask & saturated_mask
        bright_pixel_count = np.sum(led_mask)
        
        # Max brightness
        max_brightness = np.max(v_ch) / 255.0
        
        # LED is "on" if enough bright AND saturated pixels
        is_on = bright_pixel_count >= self.min_bright_pixels
        
        if is_on and np.any(led_mask):
            # Get dominant hue and saturation of LED pixels
            hue = np.median(h_ch[led_mask])
            saturation = np.median(s_ch[led_mask])
            color = self.classify_color(hue, saturation, max_brightness)
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
            saturation=saturation,
            ambient_brightness=ambient / 255.0
        )
    
    def analyze_video(self, progress_interval: int = 100) -> List[LEDState]:
        """Analyze entire video frame by frame with temporal smoothing."""
        # Calibrate first
        self.calibrate_lighting()
        
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
                print(f"   Frame {frame_num}/{self.total_frames} ({100*frame_num/self.total_frames:.1f}%)")
            
            frame_num += 1
        
        # Apply temporal smoothing to reduce noise
        self._apply_temporal_smoothing()
        
        print(f"   Analyzed {len(self.states)} frames")
        return self.states
    
    def _apply_temporal_smoothing(self):
        """
        Smooth color classification over time to reduce flickering artifacts.
        Only smooths isolated single-frame glitches, preserves intentional transitions.
        """
        if len(self.states) < 3:
            return
        
        smoothed_colors = [s.color for s in self.states]
        
        # Only fix isolated single-frame color glitches
        for i in range(1, len(self.states) - 1):
            if not self.states[i].is_on:
                continue
            
            prev_color = self.states[i-1].color if self.states[i-1].is_on else None
            curr_color = self.states[i].color
            next_color = self.states[i+1].color if self.states[i+1].is_on else None
            
            # If this frame is different from both neighbors but neighbors match,
            # it's likely a glitch - smooth it
            if prev_color and next_color and prev_color == next_color and curr_color != prev_color:
                # Check if it's truly isolated (single frame)
                if i >= 2 and self.states[i-2].is_on and self.states[i-2].color == prev_color:
                    smoothed_colors[i] = prev_color
        
        # Apply smoothed colors
        for i, color in enumerate(smoothed_colors):
            self.states[i].color = color
    
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
                if current_color is not None and state.color != current_color:
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
        color_counts = Counter(s.color for s in states)
        dominant_color = color_counts.most_common(1)[0][0]
        
        # Calculate confidence based on color consistency
        confidence = color_counts[dominant_color] / len(states)
        
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
            confidence=confidence,
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
        """Classify the LED pattern type."""
        bright_pixels = [s.bright_pixel_count for s in states]
        
        if len(bright_pixels) < 3:
            return 'blink'
        
        bp_std = np.std(bright_pixels)
        bp_mean = np.mean(bright_pixels)
        cv = bp_std / bp_mean if bp_mean > 0 else 0
        
        if cv < 0.2:
            return 'solid'
        
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
    
    def detect_sequences(self, events: List[LEDEvent]) -> List[PatternSequence]:
        """
        Group events into higher-level sequences (alternating, repeating patterns).
        """
        if len(events) < 2:
            return [self._create_single_sequence(e) for e in events]
        
        sequences = []
        i = 0
        
        while i < len(events):
            # Try to detect alternating pattern (A-B-A-B...)
            alternating = self._detect_alternating(events, i)
            if alternating:
                sequences.append(alternating)
                i += len(alternating.events)
                continue
            
            # Try to detect repeating pattern (A-A-A...)
            repeating = self._detect_repeating(events, i)
            if repeating:
                sequences.append(repeating)
                i += len(repeating.events)
                continue
            
            # Single event
            sequences.append(self._create_single_sequence(events[i]))
            i += 1
        
        return sequences
    
    def _detect_alternating(self, events: List[LEDEvent], start_idx: int, 
                           min_repeats: int = 2) -> Optional[PatternSequence]:
        """Detect alternating color patterns (e.g., GREEN-RED-GREEN-RED)."""
        if start_idx + 3 >= len(events):  # Need at least 4 events for alternating
            return None
        
        # Check for A-B pattern
        color_a = events[start_idx].color
        color_b = events[start_idx + 1].color
        
        if color_a == color_b:
            return None
        
        # Check duration similarity (within 50%)
        dur_a = events[start_idx].duration_ms
        dur_b = events[start_idx + 1].duration_ms
        
        pattern_events = [events[start_idx], events[start_idx + 1]]
        idx = start_idx + 2
        
        while idx + 1 < len(events):
            next_a = events[idx]
            next_b = events[idx + 1] if idx + 1 < len(events) else None
            
            # Check if pattern continues
            if next_a.color != color_a:
                break
            if next_b and next_b.color != color_b:
                break
            
            # Check duration similarity
            if not self._similar_duration(next_a.duration_ms, dur_a):
                break
            if next_b and not self._similar_duration(next_b.duration_ms, dur_b):
                break
            
            pattern_events.append(next_a)
            if next_b:
                pattern_events.append(next_b)
            idx += 2
        
        if len(pattern_events) < min_repeats * 2:
            return None
        
        repeat_count = len(pattern_events) // 2
        total_duration = sum(e.duration_ms for e in pattern_events)
        
        return PatternSequence(
            events=pattern_events,
            sequence_type='alternating',
            colors=[color_a, color_b],
            repeat_count=repeat_count,
            total_duration_ms=total_duration,
            description=f"{color_a.upper()} ↔ {color_b.upper()} (×{repeat_count})"
        )
    
    def _detect_repeating(self, events: List[LEDEvent], start_idx: int,
                         min_repeats: int = 2) -> Optional[PatternSequence]:
        """Detect repeating same-color patterns."""
        if start_idx >= len(events):
            return None
        
        color = events[start_idx].color
        pattern = events[start_idx].pattern
        duration = events[start_idx].duration_ms
        
        pattern_events = [events[start_idx]]
        idx = start_idx + 1
        
        while idx < len(events):
            next_event = events[idx]
            
            if next_event.color != color:
                break
            if not self._similar_duration(next_event.duration_ms, duration, tolerance=0.4):
                break
            
            pattern_events.append(next_event)
            idx += 1
        
        if len(pattern_events) < min_repeats:
            return None
        
        total_duration = sum(e.duration_ms for e in pattern_events)
        
        return PatternSequence(
            events=pattern_events,
            sequence_type='repeating',
            colors=[color],
            repeat_count=len(pattern_events),
            total_duration_ms=total_duration,
            description=f"{color.upper()} {pattern} (×{len(pattern_events)})"
        )
    
    def _create_single_sequence(self, event: LEDEvent) -> PatternSequence:
        """Create a sequence from a single event."""
        return PatternSequence(
            events=[event],
            sequence_type='single',
            colors=[event.color],
            repeat_count=1,
            total_duration_ms=event.duration_ms,
            description=f"{event.color.upper()} {event.pattern} ({self._format_duration(event.duration_ms)})"
        )
    
    def _similar_duration(self, dur1: float, dur2: float, tolerance: float = 0.5) -> bool:
        """Check if two durations are similar within tolerance."""
        if dur2 == 0:
            return False
        ratio = dur1 / dur2
        return (1 - tolerance) <= ratio <= (1 + tolerance)
    
    def _format_duration(self, ms: float) -> str:
        """Format duration in human-readable form."""
        if ms >= 1000:
            return f"{ms/1000:.1f}s"
        return f"{ms:.0f}ms"
    
    def generate_formatted_timeline(self, events: List[LEDEvent], 
                                    sequences: List[PatternSequence]) -> str:
        """Generate human-readable timeline in the requested format."""
        lines = []
        
        for seq in sequences:
            if seq.sequence_type == 'alternating':
                # Format: → COLOR1 pattern (dur) ↔ COLOR2 pattern (dur) ×N
                e1, e2 = seq.events[0], seq.events[1]
                line = f"→ {e1.color.upper()} {e1.pattern} ({self._format_duration(e1.duration_ms)}) "
                line += f"↔ {e2.color.upper()} {e2.pattern} ({self._format_duration(e2.duration_ms)}) "
                line += f"  ← alternating ×{seq.repeat_count}"
                lines.append(line)
            elif seq.sequence_type == 'repeating' and seq.repeat_count > 1:
                e = seq.events[0]
                line = f"→ {e.color.upper()} {e.pattern} ({self._format_duration(e.duration_ms)}) ×{seq.repeat_count}"
                lines.append(line)
            else:
                # Single event
                e = seq.events[0]
                line = f"→ {e.color.upper()} {e.pattern} ({self._format_duration(e.duration_ms)})"
                lines.append(line)
        
        return "\n".join(lines)
    
    def generate_report(self) -> AnalysisResult:
        """Generate complete analysis report with sequences."""
        events = self.detect_events()
        sequences = self.detect_sequences(events)
        formatted_timeline = self.generate_formatted_timeline(events, sequences)
        
        # Color summary
        color_summary = {}
        color_durations = {}
        for event in events:
            color_summary[event.color] = color_summary.get(event.color, 0) + 1
            color_durations[event.color] = color_durations.get(event.color, 0) + event.duration_ms
        
        # Pattern summary
        pattern_summary = {}
        for event in events:
            pattern_summary[event.pattern] = pattern_summary.get(event.pattern, 0) + 1
        
        # Timeline
        timeline = []
        for event in events:
            timeline.append({
                'start_sec': round(event.start_ms / 1000, 3),
                'end_sec': round(event.end_ms / 1000, 3),
                'duration_ms': round(event.duration_ms, 1),
                'color': event.color,
                'pattern': event.pattern,
                'brightness': round(event.avg_brightness, 2),
                'confidence': round(event.confidence, 2)
            })
        
        return AnalysisResult(
            video_path=self.video_path,
            fps=self.fps,
            total_frames=self.total_frames,
            duration_ms=self.duration_ms,
            led_roi=self.led_roi,
            lighting_profile=self.lighting_profile,
            events=events,
            sequences=sequences,
            color_summary=color_summary,
            pattern_summary=pattern_summary,
            color_durations=color_durations,
            timeline=timeline,
            formatted_timeline=formatted_timeline
        )
    
    def print_report(self, result: AnalysisResult):
        """Print human-readable report."""
        print("\n" + "=" * 70)
        print("LED PATTERN ANALYSIS REPORT v3")
        print("=" * 70)
        
        print(f"\n📹 Video: {result.video_path}")
        print(f"   Duration: {result.duration_ms/1000:.2f} seconds")
        print(f"   FPS: {result.fps:.1f}")
        print(f"   Total Frames: {result.total_frames}")
        print(f"   LED ROI: x={result.led_roi[0]}, y={result.led_roi[1]}, "
              f"w={result.led_roi[2]}, h={result.led_roi[3]}")
        
        print(f"\n🔆 Lighting Profile:")
        print(f"   Condition: {result.lighting_profile.get('condition', 'unknown')}")
        print(f"   Avg brightness: {result.lighting_profile.get('avg_brightness', 0):.1f}")
        print(f"   Adaptive threshold: {result.lighting_profile.get('bright_threshold', 200)}")
        
        print(f"\n{'─' * 70}")
        print("📊 SUMMARY")
        print(f"{'─' * 70}")
        print(f"Total LED events: {len(result.events)}")
        print(f"Pattern sequences: {len(result.sequences)}")
        
        print(f"\n🎨 Colors detected:")
        for color, count in sorted(result.color_summary.items(), key=lambda x: -x[1]):
            duration = result.color_durations.get(color, 0)
            print(f"   {color:10s}: {count:3d} events, total {duration/1000:.2f}s")
        
        print(f"\n⚡ Patterns detected:")
        for pattern, count in sorted(result.pattern_summary.items(), key=lambda x: -x[1]):
            print(f"   {pattern:10s}: {count:3d} events")
        
        print(f"\n{'─' * 70}")
        print("📅 FORMATTED TIMELINE")
        print(f"{'─' * 70}")
        print(result.formatted_timeline)
        
        print(f"\n{'─' * 70}")
        print("📈 DETAILED EVENTS")
        print(f"{'─' * 70}")
        
        color_emoji = {
            'red': '🔴', 'orange': '🟠', 'yellow': '🟡', 'green': '🟢',
            'cyan': '🔵', 'blue': '🔵', 'magenta': '🟣', 'white': '⚪'
        }
        
        for i, event in enumerate(result.events, 1):
            emoji = color_emoji.get(event.color, '⬜')
            conf = f"[{event.confidence:.0%}]" if event.confidence < 1.0 else ""
            print(f"{emoji} {i:2d}. {event.color.upper():8s} {event.pattern:8s} "
                  f"{event.start_ms/1000:6.2f}s → {event.end_ms/1000:6.2f}s "
                  f"({self._format_duration(event.duration_ms):>6s}) {conf}")
        
        print("\n" + "=" * 70)
    
    def export_json(self, result: AnalysisResult, output_path: str):
        """Export analysis to JSON file."""
        data = {
            'video_path': result.video_path,
            'fps': result.fps,
            'total_frames': result.total_frames,
            'duration_ms': result.duration_ms,
            'led_roi': list(result.led_roi),
            'lighting_profile': result.lighting_profile,
            'summary': {
                'total_events': len(result.events),
                'total_sequences': len(result.sequences),
                'colors': result.color_summary,
                'patterns': result.pattern_summary,
                'color_durations_ms': result.color_durations
            },
            'formatted_timeline': result.formatted_timeline,
            'sequences': [
                {
                    'type': seq.sequence_type,
                    'colors': seq.colors,
                    'repeat_count': seq.repeat_count,
                    'total_duration_ms': seq.total_duration_ms,
                    'description': seq.description
                }
                for seq in result.sequences
            ],
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
                    'confidence': round(e.confidence, 3),
                    'rise_time_ms': round(e.rise_time_ms, 1) if e.rise_time_ms else None,
                    'fall_time_ms': round(e.fall_time_ms, 1) if e.fall_time_ms else None
                }
                for i, e in enumerate(result.events)
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Exported to: {output_path}")
    
    def generate_pattern_string(self, events: List[LEDEvent]) -> str:
        """Generate compact pattern string like 'GREEN×5(137ms) → GREEN(512ms) → RED(307ms)'"""
        if not events:
            return ""
        
        groups = []
        i = 0
        while i < len(events):
            color = events[i].color.upper()
            duration = events[i].duration_ms
            count = 1
            
            # Count consecutive same-color, similar-duration events
            j = i + 1
            while j < len(events):
                if events[j].color.upper() != color:
                    break
                # Similar duration = within 50%
                if not (0.5 <= events[j].duration_ms / duration <= 1.5):
                    break
                count += 1
                j += 1
            
            # Format duration
            if duration >= 1000:
                dur_str = f"{duration/1000:.1f}s"
            else:
                dur_str = f"{int(duration)}ms"
            
            if count > 1:
                groups.append(f"{color}×{count}({dur_str})")
            else:
                groups.append(f"{color}({dur_str})")
            
            i = j
        
        return " → ".join(groups)
    
    def export_minimal_json(self, result: AnalysisResult, output_path: str):
        """Export minimal, easy-to-extract JSON format."""
        events = result.events
        pattern_string = self.generate_pattern_string(events)
        
        data = {
            "sequence": [e.color for e in events],
            "pattern": pattern_string,
            "duration_sec": round(result.duration_ms / 1000, 2),
            "events": [
                {
                    "t": round(e.start_ms / 1000, 2),
                    "color": e.color,
                    "ms": int(round(e.duration_ms))
                }
                for e in events
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Exported (minimal): {output_path}")
        print(f"   Pattern: {pattern_string}")
    
    def create_annotated_video(self, result: AnalysisResult, output_path: str):
        """Create video with LED state annotations."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
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
        
        for state in self.states:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, state.frame_num)
            ret, frame = self.cap.read()
            if not ret:
                break
            
            x, y, w, h = self.led_roi
            color = color_bgr.get(state.color, (128, 128, 128))
            thickness = 3 if state.is_on else 1
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            status = f"{state.color.upper()}" if state.is_on else "OFF"
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, color, 2)
            
            time_text = f"T: {state.timestamp_ms/1000:.2f}s"
            cv2.putText(frame, time_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)
            
            # Show ambient level
            ambient_text = f"Ambient: {state.ambient_brightness:.2f}"
            cv2.putText(frame, ambient_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 2)
            
            px_text = f"Bright px: {state.bright_pixel_count}"
            cv2.putText(frame, px_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"\n🎬 Annotated video: {output_path}")
    
    def close(self):
        """Release video capture."""
        self.cap.release()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze LED patterns in video')
    parser.add_argument('video', help='Path to input video')
    parser.add_argument('--output', '-o', help='Output JSON path')
    parser.add_argument('--minimal', '-m', action='store_true',
                        help='Output minimal JSON (easy to extract)')
    parser.add_argument('--annotate', '-a', action='store_true',
                        help='Create annotated video')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress detailed output, just show pattern')
    parser.add_argument('--roi', help='Manual ROI as x,y,w,h')
    parser.add_argument('--threshold', '-t', type=int, default=200,
                        help='Min bright pixels for LED on (default: 200)')
    
    args = parser.parse_args()
    
    analyzer = LEDAnalyzer(args.video)
    analyzer.BASE_MIN_BRIGHT_PIXELS = args.threshold
    
    if args.roi:
        x, y, w, h = map(int, args.roi.split(','))
        analyzer.set_roi(x, y, w, h)
    
    analyzer.analyze_video()
    result = analyzer.generate_report()
    
    # Print report (unless quiet mode)
    if not args.quiet:
        analyzer.print_report(result)
    else:
        # In quiet mode, just print the pattern
        pattern = analyzer.generate_pattern_string(result.events)
        print(f"\n{pattern}\n")
    
    # Determine output path
    video_path = Path(args.video)
    output_path = args.output if args.output else str(video_path.with_suffix('.json'))
    
    # Export JSON (minimal or full)
    if args.minimal:
        analyzer.export_minimal_json(result, output_path)
    else:
        analyzer.export_json(result, output_path)
    
    # Create annotated video if requested
    if args.annotate:
        annotated_path = video_path.parent / f"{video_path.stem}_annotated.mp4"
        analyzer.create_annotated_video(result, str(annotated_path))
    
    analyzer.close()


if __name__ == '__main__':
    main()
