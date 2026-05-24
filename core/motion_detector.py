"""
core/motion_detector.py — Motion detection and analysis for video content.

Provides frame differencing, motion energy computation, moving object
detection, optical flow analysis, motion heatmaps, camera motion
detection, and steadiness scoring. Uses PIL/numpy for core operations
with OpenCV acceleration when available.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from utils.ffmpeg_utils import probe_video
from utils.logger import get_logger

logger = get_logger("motion_detector")


# ── Data Classes ──────────────────────────────────────────

@dataclass
class MotionFrame:
    """Motion magnitude for a single frame pair."""

    timestamp: float
    magnitude: float  # 0-1, normalized motion magnitude
    pixel_count: int = 0  # Number of pixels exceeding threshold
    frame_index: int = 0


@dataclass
class MovingObject:
    """A detected moving object region."""

    x: int
    y: int
    width: int
    height: int
    area: int  # Area in pixels
    magnitude: float = 0.0  # Average motion magnitude
    timestamp: float = 0.0

    @property
    def center_x(self) -> int:
        """X coordinate of object center."""
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        """Y coordinate of object center."""
        return self.y + self.height // 2


@dataclass
class OpticalFlowResult:
    """Result of dense optical flow analysis."""

    flow_magnitudes: list[float] = field(default_factory=list)  # Per-frame average
    flow_directions: list[float] = field(default_factory=list)  # Per-frame average direction (radians)
    timestamps: list[float] = field(default_factory=list)
    average_magnitude: float = 0.0
    dominant_direction: float = 0.0  # Radians
    is_available: bool = False  # Whether optical flow was computed


@dataclass
class CameraMotionResult:
    """Detected camera motion (pan/zoom/shake)."""

    has_camera_motion: bool = False
    motion_type: str = "static"  # "static", "pan_left", "pan_right", "zoom_in", "zoom_out", "shake"
    motion_magnitude: float = 0.0  # 0-1
    confidence: float = 0.0
    pan_direction: str = "none"  # "left", "right", "none"
    pan_speed: float = 0.0  # Pixels per frame
    zoom_type: str = "none"  # "in", "out", "none"
    zoom_speed: float = 0.0  # Scale factor per frame
    shake_amount: float = 0.0  # 0-1, how shaky


# ── MotionDetector Class ──────────────────────────────────

class MotionDetector:
    """Motion detection and analysis for video content.

    Provides frame differencing using PIL/numpy (no OpenCV required),
    optical flow using OpenCV when available, motion heatmaps for
    crop position optimization, and camera motion detection.
    """

    def __init__(self, downscale_width: int = 320) -> None:
        """Initialize the motion detector.

        Args:
            downscale_width: Width to downscale frames to for faster
                processing. Height is computed to maintain aspect ratio.
        """
        self.downscale_width = downscale_width
        self._cv2 = None

    def _get_cv2(self):
        """Lazily import OpenCV."""
        if self._cv2 is None:
            try:
                import cv2
                self._cv2 = cv2
            except ImportError:
                logger.debug("OpenCV not available; using frame differencing fallback")
        return self._cv2

    def _extract_frame_pil(self, video_path: Path, timestamp: float) -> np.ndarray | None:
        """Extract a single frame from a video using FFmpeg and PIL.

        Args:
            video_path: Path to the video file.
            timestamp: Time in seconds to extract the frame at.

        Returns:
            Grayscale numpy array of the frame, or None on failure.
        """
        cmd: list[str] = [
            "ffmpeg",
            "-ss", f"{timestamp:.3f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-vf", f"scale={self.downscale_width}:-1",
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "png",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=30,
            )
            if result.returncode != 0 or len(result.stdout) < 100:
                return None

            from io import BytesIO
            img = Image.open(BytesIO(result.stdout))
            return np.array(img.convert("L"), dtype=np.float64)
        except Exception as exc:
            logger.debug("Frame extraction failed at %.3fs: %s", timestamp, exc)
            return None

    def _extract_frame_raw(self, video_path: Path, timestamp: float, width: int, height: int) -> np.ndarray | None:
        """Extract a single frame as raw grayscale pixels using FFmpeg.

        Faster than PIL-based extraction since it avoids PNG encoding.

        Args:
            video_path: Path to the video file.
            timestamp: Time in seconds.
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            Grayscale numpy array, or None on failure.
        """
        cmd: list[str] = [
            "ffmpeg",
            "-ss", f"{timestamp:.3f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-vf", f"scale={width}:{height}",
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                return None

            expected_size = width * height
            if len(result.stdout) < expected_size:
                return None

            return np.frombuffer(result.stdout[:expected_size], dtype=np.uint8).reshape(height, width).astype(np.float64)
        except Exception as exc:
            logger.debug("Raw frame extraction failed at %.3fs: %s", timestamp, exc)
            return None

    def _get_frame_dimensions(self, video_path: Path) -> tuple[int, int]:
        """Get downscaled frame dimensions based on video aspect ratio.

        Args:
            video_path: Path to the video file.

        Returns:
            Tuple of (width, height) for downscaled frames.
        """
        try:
            info = probe_video(video_path)
            aspect = info.width / max(info.height, 1)
            width = self.downscale_width
            height = max(1, int(width / aspect))
            # Ensure even dimensions
            height = height + (height % 2)
            return width, height
        except Exception:
            return self.downscale_width, int(self.downscale_width * 9 / 16)

    def compute_frame_difference(
        self,
        video_path: Path,
        sample_rate: float = 1.0,
    ) -> list[MotionFrame]:
        """Compute frame-by-frame motion magnitude using frame differencing.

        Extracts frames at the specified sample rate and computes the
        absolute pixel difference between consecutive frames.

        Args:
            video_path: Path to the video file.
            sample_rate: Frames per second to sample. Default 1.0.

        Returns:
            List of MotionFrame objects with motion magnitude per frame.

        Raises:
            FileNotFoundError: If video file doesn't exist.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        width, height = self._get_frame_dimensions(video_path)
        video_info = probe_video(video_path)
        total_duration = video_info.duration

        if total_duration <= 0:
            return []

        # Generate timestamps
        timestamps: list[float] = []
        t = 0.0
        while t < total_duration:
            timestamps.append(t)
            t += 1.0 / sample_rate

        motion_frames: list[MotionFrame] = []
        prev_frame: np.ndarray | None = None
        max_possible_diff = 255.0  # Maximum pixel difference

        for idx, ts in enumerate(timestamps):
            frame = self._extract_frame_raw(video_path, ts, width, height)

            if frame is None:
                motion_frames.append(MotionFrame(
                    timestamp=ts, magnitude=0.0, pixel_count=0, frame_index=idx
                ))
                continue

            if prev_frame is not None:
                # Ensure same dimensions
                if frame.shape != prev_frame.shape:
                    min_h = min(frame.shape[0], prev_frame.shape[0])
                    min_w = min(frame.shape[1], prev_frame.shape[1])
                    frame = frame[:min_h, :min_w]
                    prev_frame = prev_frame[:min_h, :min_w]

                # Compute absolute difference
                diff = np.abs(frame - prev_frame)

                # Threshold to remove noise
                threshold = 15.0
                significant_diff = diff[diff > threshold]

                # Normalized magnitude
                magnitude = float(np.mean(diff)) / max_possible_diff
                pixel_count = int(len(significant_diff))

                motion_frames.append(MotionFrame(
                    timestamp=ts,
                    magnitude=magnitude,
                    pixel_count=pixel_count,
                    frame_index=idx,
                ))
            else:
                motion_frames.append(MotionFrame(
                    timestamp=ts, magnitude=0.0, pixel_count=0, frame_index=idx
                ))

            prev_frame = frame.copy()

        logger.info(
            "Frame difference: %d frames, avg magnitude=%.4f",
            len(motion_frames),
            sum(m.magnitude for m in motion_frames) / max(len(motion_frames), 1),
        )

        return motion_frames

    def compute_motion_energy(
        self,
        video_path: Path,
        sample_interval: float = 2.0,
    ) -> np.ndarray:
        """Compute motion energy per time interval.

        Averages frame difference magnitudes within each sample interval
        to produce a motion energy signal compatible with the analyzer.

        Args:
            video_path: Path to the video file.
            sample_interval: Seconds between energy samples.

        Returns:
            Normalized motion energy array (0-1).

        Raises:
            FileNotFoundError: If video file doesn't exist.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_info = probe_video(video_path)
        total_duration = video_info.duration

        if total_duration <= 0:
            return np.array([])

        num_samples = max(1, int(total_duration / sample_interval))
        width, height = self._get_frame_dimensions(video_path)

        # Compute motion at each sample interval
        motion_values = np.zeros(num_samples, dtype=float)
        max_possible_diff = 255.0

        prev_frame: np.ndarray | None = None

        for i in range(num_samples):
            timestamp = i * sample_interval + sample_interval * 0.5
            frame = self._extract_frame_raw(video_path, timestamp, width, height)

            if frame is None:
                continue

            if prev_frame is not None:
                # Ensure same dimensions
                if frame.shape != prev_frame.shape:
                    min_h = min(frame.shape[0], prev_frame.shape[0])
                    min_w = min(frame.shape[1], prev_frame.shape[1])
                    frame = frame[:min_h, :min_w]
                    prev_frame = prev_frame[:min_h, :min_w]

                diff = np.abs(frame - prev_frame)
                motion_values[i] = float(np.mean(diff)) / max_possible_diff

            prev_frame = frame.copy() if frame is not None else None

        # Normalize to 0-1
        max_val = np.max(motion_values)
        if max_val > 1e-6:
            motion_values = motion_values / max_val

        logger.info(
            "Motion energy: %d samples over %.1fs",
            num_samples, total_duration,
        )

        return motion_values

    def detect_moving_objects(
        self,
        video_path: Path,
        min_area: int = 500,
        timestamp: float = 0.0,
    ) -> list[MovingObject]:
        """Detect significant motion regions in a frame.

        Compares the frame at the given timestamp with a frame slightly
        earlier, identifies regions with significant pixel differences,
        and returns bounding boxes for moving objects.

        Args:
            video_path: Path to the video file.
            min_area: Minimum area in pixels for a region to be
                considered a moving object. Default 500.
            timestamp: Time in seconds to analyze.

        Returns:
            List of MovingObject bounding boxes.

        Raises:
            FileNotFoundError: If video file doesn't exist.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        width, height = self._get_frame_dimensions(video_path)
        prev_ts = max(0.0, timestamp - 0.5)

        # Extract current and previous frames
        current_frame = self._extract_frame_raw(video_path, timestamp, width, height)
        prev_frame = self._extract_frame_raw(video_path, prev_ts, width, height)

        if current_frame is None or prev_frame is None:
            return []

        # Ensure same dimensions
        if current_frame.shape != prev_frame.shape:
            return []

        # Compute difference
        diff = np.abs(current_frame - prev_frame)

        # Threshold to binary
        threshold = 25.0
        binary = (diff > threshold).astype(np.uint8)

        # Find connected components using simple flood fill approach
        objects = self._find_connected_components(binary, min_area)

        # Scale object coordinates back to original video dimensions
        video_info = probe_video(video_path)
        scale_x = video_info.width / max(width, 1)
        scale_y = video_info.height / max(height, 1)

        moving_objects: list[MovingObject] = []
        for obj in objects:
            # Compute average motion magnitude in the object region
            y1, y2 = max(0, obj["y"]), min(binary.shape[0], obj["y"] + obj["h"])
            x1, x2 = max(0, obj["x"]), min(binary.shape[1], obj["x"] + obj["w"])
            region_diff = diff[y1:y2, x1:x2]
            avg_magnitude = float(np.mean(region_diff)) / 255.0 if region_diff.size > 0 else 0.0

            moving_objects.append(MovingObject(
                x=int(obj["x"] * scale_x),
                y=int(obj["y"] * scale_y),
                width=int(obj["w"] * scale_x),
                height=int(obj["h"] * scale_y),
                area=int(obj["area"] * scale_x * scale_y),
                magnitude=avg_magnitude,
                timestamp=timestamp,
            ))

        return moving_objects

    @staticmethod
    def _find_connected_components(
        binary: np.ndarray, min_area: int
    ) -> list[dict[str, int]]:
        """Find connected components in a binary image using simple labeling.

        Uses a two-pass connected component labeling algorithm.

        Args:
            binary: Binary image (0 or 1).
            min_area: Minimum component area in pixels.

        Returns:
            List of dicts with x, y, w, h, area for each component.
        """
        if binary.size == 0:
            return []

        h, w = binary.shape
        labels = np.zeros((h, w), dtype=np.int32)
        current_label = 0
        components: dict[int, list[tuple[int, int]]] = {}

        # Simple flood-fill based labeling
        for y in range(h):
            for x in range(w):
                if binary[y, x] == 1 and labels[y, x] == 0:
                    current_label += 1
                    # Flood fill
                    stack: list[tuple[int, int]] = [(y, x)]
                    pixels: list[tuple[int, int]] = []

                    while stack:
                        cy, cx = stack.pop()
                        if cy < 0 or cy >= h or cx < 0 or cx >= w:
                            continue
                        if binary[cy, cx] != 1 or labels[cy, cx] != 0:
                            continue

                        labels[cy, cx] = current_label
                        pixels.append((cy, cx))

                        # 4-connectivity
                        stack.extend([(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)])

                    if len(pixels) >= min_area:
                        components[current_label] = pixels

        # Convert to bounding boxes
        result: list[dict[str, int]] = []
        for label_id, pixels in components.items():
            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            result.append({
                "x": min_x,
                "y": min_y,
                "w": max_x - min_x + 1,
                "h": max_y - min_y + 1,
                "area": len(pixels),
            })

        return result

    def compute_optical_flow(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
    ) -> OpticalFlowResult:
        """Compute dense optical flow analysis.

        Uses OpenCV's Farneback dense optical flow when available,
        falling back to frame differencing magnitude estimation.

        Args:
            video_path: Path to the video file.
            start_time: Start time in seconds.
            end_time: End time in seconds.

        Returns:
            OpticalFlowResult with flow magnitudes and directions.

        Raises:
            FileNotFoundError: If video file doesn't exist.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cv2 = self._get_cv2()

        if cv2 is not None:
            return self._compute_optical_flow_cv2(video_path, start_time, end_time, cv2)
        else:
            return self._compute_optical_flow_fallback(video_path, start_time, end_time)

    def _compute_optical_flow_cv2(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        cv2,
    ) -> OpticalFlowResult:
        """Compute optical flow using OpenCV Farneback method.

        Args:
            video_path: Path to the video file.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            cv2: OpenCV module.

        Returns:
            OpticalFlowResult with flow data.
        """
        width, height = self._get_frame_dimensions(video_path)
        sample_interval = 1.0  # 1 FPS for optical flow

        timestamps: list[float] = []
        magnitudes: list[float] = []
        directions: list[float] = []

        prev_gray: np.ndarray | None = None
        t = start_time

        while t <= end_time:
            # Extract frame using FFmpeg
            frame = self._extract_frame_raw(video_path, t, width, height)

            if frame is None:
                t += sample_interval
                continue

            gray = frame.astype(np.uint8)

            if prev_gray is not None and prev_gray.shape == gray.shape:
                # Compute dense optical flow using Farneback
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )

                # Compute magnitude and direction
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                avg_magnitude = float(np.mean(mag))
                avg_direction = float(np.mean(ang))

                # Normalize magnitude
                normalized_mag = min(1.0, avg_magnitude / 50.0)

                magnitudes.append(normalized_mag)
                directions.append(avg_direction)
                timestamps.append(t)

            prev_gray = gray
            t += sample_interval

        # Compute aggregate statistics
        avg_mag = sum(magnitudes) / len(magnitudes) if magnitudes else 0.0

        # Compute dominant direction using circular mean
        dominant_dir = 0.0
        if directions:
            sin_sum = sum(math.sin(d) for d in directions)
            cos_sum = sum(math.cos(d) for d in directions)
            dominant_dir = math.atan2(sin_sum, cos_sum)

        return OpticalFlowResult(
            flow_magnitudes=magnitudes,
            flow_directions=directions,
            timestamps=timestamps,
            average_magnitude=round(avg_mag, 4),
            dominant_direction=round(dominant_dir, 4),
            is_available=True,
        )

    def _compute_optical_flow_fallback(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
    ) -> OpticalFlowResult:
        """Fallback optical flow estimation using frame differencing.

        When OpenCV is not available, estimates motion magnitude
        from frame differences and direction from horizontal
        asymmetry of the difference image.

        Args:
            video_path: Path to the video file.
            start_time: Start time in seconds.
            end_time: End time in seconds.

        Returns:
            OpticalFlowResult with estimated flow data.
        """
        width, height = self._get_frame_dimensions(video_path)
        sample_interval = 1.0

        timestamps: list[float] = []
        magnitudes: list[float] = []
        directions: list[float] = []

        prev_frame: np.ndarray | None = None
        t = start_time

        while t <= end_time:
            frame = self._extract_frame_raw(video_path, t, width, height)

            if frame is not None and prev_frame is not None:
                if frame.shape == prev_frame.shape:
                    diff = np.abs(frame - prev_frame)
                    mag = float(np.mean(diff)) / 255.0

                    # Estimate direction from left-right asymmetry
                    left_diff = np.mean(diff[:, :diff.shape[1] // 2])
                    right_diff = np.mean(diff[:, diff.shape[1] // 2:])
                    if abs(left_diff - right_diff) > 1e-6:
                        direction = math.pi if left_diff > right_diff else 0.0
                    else:
                        direction = 0.0

                    magnitudes.append(mag)
                    directions.append(direction)
                    timestamps.append(t)

            if frame is not None:
                prev_frame = frame.copy()
            t += sample_interval

        avg_mag = sum(magnitudes) / len(magnitudes) if magnitudes else 0.0

        return OpticalFlowResult(
            flow_magnitudes=magnitudes,
            flow_directions=directions,
            timestamps=timestamps,
            average_magnitude=round(avg_mag, 4),
            dominant_direction=0.0,
            is_available=False,  # Not real optical flow
        )

    def get_motion_heatmap(
        self,
        video_path: Path,
        num_frames: int = 50,
    ) -> np.ndarray:
        """Generate a 2D heatmap of where motion occurs in the video.

        Accumulates absolute frame differences across multiple frames
        to identify regions of consistent motion. Useful for determining
        where the subject is most likely to appear.

        Args:
            video_path: Path to the video file.
            num_frames: Number of frames to sample.

        Returns:
            2D numpy array (height x width) with motion intensity (0-1).

        Raises:
            FileNotFoundError: If video file doesn't exist.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_info = probe_video(video_path)
        total_duration = video_info.duration

        if total_duration <= 0:
            return np.array([])

        width, height = self._get_frame_dimensions(video_path)
        heatmap = np.zeros((height, width), dtype=np.float64)

        # Sample timestamps evenly
        timestamps = np.linspace(0, total_duration, num_frames + 2)[1:-1].tolist()

        prev_frame: np.ndarray | None = None

        for ts in timestamps:
            frame = self._extract_frame_raw(video_path, ts, width, height)

            if frame is None:
                continue

            if prev_frame is not None and frame.shape == prev_frame.shape:
                diff = np.abs(frame - prev_frame)
                heatmap += diff

            prev_frame = frame.copy()

        # Normalize heatmap to 0-1
        max_val = np.max(heatmap)
        if max_val > 1e-6:
            heatmap = heatmap / max_val

        # Apply Gaussian-like smoothing (simple box blur repeated)
        heatmap = self._blur_heatmap(heatmap, kernel_size=5)

        logger.info(
            "Motion heatmap: %dx%d from %d frames",
            height, width, len(timestamps),
        )

        return heatmap

    @staticmethod
    def _blur_heatmap(heatmap: np.ndarray, kernel_size: int = 5, iterations: int = 2) -> np.ndarray:
        """Apply simple box blur to a heatmap.

        Args:
            heatmap: 2D array to blur.
            kernel_size: Size of the blur kernel.
            iterations: Number of blur passes.

        Returns:
            Blurred heatmap.
        """
        result = heatmap.copy()
        pad = kernel_size // 2

        for _ in range(iterations):
            padded = np.pad(result, pad, mode="edge")
            result = np.zeros_like(heatmap)
            for dy in range(kernel_size):
                for dx in range(kernel_size):
                    result += padded[dy:dy + heatmap.shape[0], dx:dx + heatmap.shape[1]]
            result /= (kernel_size * kernel_size)

        return result

    def detect_camera_motion(self, video_path: Path) -> CameraMotionResult:
        """Detect camera motion: pan, zoom, and shake.

        Analyzes the global motion pattern across the video to determine
        if the camera is panning left/right, zooming in/out, or shaking.
        Uses the dominant flow direction and spatial distribution of motion.

        Args:
            video_path: Path to the video file.

        Returns:
            CameraMotionResult with detected motion type and parameters.

        Raises:
            FileNotFoundError: If video file doesn't exist.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_info = probe_video(video_path)
        total_duration = video_info.duration

        if total_duration <= 2.0:
            return CameraMotionResult(motion_type="static")

        # Sample frames for camera motion analysis
        sample_count = min(30, int(total_duration))
        timestamps = np.linspace(0.5, total_duration - 0.5, sample_count).tolist()

        width, height = self._get_frame_dimensions(video_path)

        # Compute frame-to-frame shifts
        horizontal_shifts: list[float] = []
        vertical_shifts: list[float] = []
        scale_changes: list[float] = []

        prev_frame: np.ndarray | None = None

        for ts in timestamps:
            frame = self._extract_frame_raw(video_path, ts, width, height)
            if frame is None or prev_frame is None or frame.shape != prev_frame.shape:
                if frame is not None:
                    prev_frame = frame.copy()
                continue

            # Compute phase correlation to estimate global shift
            h_shift, v_shift, scale_change = self._estimate_global_motion(prev_frame, frame)

            horizontal_shifts.append(h_shift)
            vertical_shifts.append(v_shift)
            scale_changes.append(scale_change)

            prev_frame = frame.copy()

        if not horizontal_shifts:
            return CameraMotionResult(motion_type="static")

        # Analyze shifts to determine camera motion type
        avg_h_shift = np.mean(horizontal_shifts)
        avg_v_shift = np.mean(vertical_shifts)
        avg_scale = np.mean(scale_changes)

        # Compute shake (high-frequency variation in shifts)
        if len(horizontal_shifts) >= 3:
            h_variation = float(np.std(horizontal_shifts))
            v_variation = float(np.std(vertical_shifts))
            shake_amount = min(1.0, (h_variation + v_variation) / 20.0)
        else:
            shake_amount = 0.0

        # Determine motion type
        motion_type = "static"
        pan_direction = "none"
        pan_speed = 0.0
        zoom_type = "none"
        zoom_speed = 0.0

        # Pan detection: consistent horizontal shift
        pan_threshold = 2.0  # pixels per frame
        if abs(avg_h_shift) > pan_threshold:
            pan_direction = "left" if avg_h_shift < 0 else "right"
            pan_speed = abs(avg_h_shift)
            motion_type = f"pan_{pan_direction}"

        # Zoom detection: consistent scale change
        zoom_threshold = 0.005  # scale factor per frame
        if abs(avg_scale) > zoom_threshold:
            zoom_type = "in" if avg_scale > 0 else "out"
            zoom_speed = abs(avg_scale)
            if motion_type == "static":
                motion_type = f"zoom_{zoom_type}"
            else:
                motion_type = f"{motion_type}_zoom_{zoom_type}"

        # Shake detection
        if shake_amount > 0.3:
            if motion_type == "static":
                motion_type = "shake"
            else:
                motion_type = f"{motion_type}_shake"

        # Overall motion magnitude
        motion_magnitude = min(1.0, (abs(avg_h_shift) + abs(avg_v_shift)) / 30.0 + shake_amount)

        has_camera_motion = motion_type != "static"

        result = CameraMotionResult(
            has_camera_motion=has_camera_motion,
            motion_type=motion_type,
            motion_magnitude=round(motion_magnitude, 3),
            confidence=round(min(1.0, len(horizontal_shifts) / 10.0), 2),
            pan_direction=pan_direction,
            pan_speed=round(pan_speed, 2),
            zoom_type=zoom_type,
            zoom_speed=round(zoom_speed, 4),
            shake_amount=round(shake_amount, 3),
        )

        logger.info(
            "Camera motion: type=%s, magnitude=%.3f, pan=%s@%.1fpx, zoom=%s@%.4f, shake=%.3f",
            motion_type, motion_magnitude, pan_direction, pan_speed,
            zoom_type, zoom_speed, shake_amount,
        )

        return result

    @staticmethod
    def _estimate_global_motion(
        prev_frame: np.ndarray, current_frame: np.ndarray
    ) -> tuple[float, float, float]:
        """Estimate global motion between two frames using block matching.

        Compares the center region of both frames to estimate the
        horizontal shift, vertical shift, and scale change.

        Args:
            prev_frame: Previous frame as grayscale array.
            current_frame: Current frame as grayscale array.

        Returns:
            Tuple of (h_shift, v_shift, scale_change).
        """
        h, w = prev_frame.shape
        if h < 20 or w < 20:
            return 0.0, 0.0, 0.0

        # Use center 50% of the frame for motion estimation
        margin_y = h // 4
        margin_x = w // 4
        prev_center = prev_frame[margin_y:h - margin_y, margin_x:w - margin_x]
        curr_center = current_frame[margin_y:h - margin_y, margin_x:w - margin_x]

        ch, cw = prev_center.shape

        # Search for best match in a small window
        best_shift_x = 0.0
        best_shift_y = 0.0
        best_error = float("inf")

        search_range = 15  # pixels

        for dy in range(-search_range, search_range + 1, 2):
            for dx in range(-search_range, search_range + 1, 2):
                # Shift current frame
                y1_src = max(0, dy)
                y2_src = min(ch, ch + dy)
                x1_src = max(0, dx)
                x2_src = min(cw, cw + dx)

                y1_dst = max(0, -dy)
                y2_dst = min(ch, ch - dy)
                x1_dst = max(0, -dx)
                x2_dst = min(cw, cw - dx)

                if y2_src <= y1_src or x2_src <= x1_src:
                    continue
                if y2_dst <= y1_dst or x2_dst <= x1_dst:
                    continue

                region_prev = prev_center[y1_dst:y2_dst, x1_dst:x2_dst]
                region_curr = curr_center[y1_src:y2_src, x1_src:x2_src]

                if region_prev.shape != region_curr.shape:
                    continue

                error = float(np.sum(np.abs(region_prev - region_curr)))

                if error < best_error:
                    best_error = error
                    best_shift_x = float(dx)
                    best_shift_y = float(dy)

        # Estimate scale change by comparing corner regions
        corner_size = min(20, ch // 4, cw // 4)
        if corner_size < 5:
            return best_shift_x, best_shift_y, 0.0

        # Compare top-left and bottom-right corners
        tl_prev = prev_center[:corner_size, :corner_size]
        br_prev = prev_center[-corner_size:, -corner_size:]

        tl_curr = curr_center[:corner_size, :corner_size]
        br_curr = curr_center[-corner_size:, -corner_size:]

        # Scale change: if bottom-right diff > top-left diff, zooming in
        tl_diff = float(np.mean(np.abs(tl_prev - tl_curr)))
        br_diff = float(np.mean(np.abs(br_prev - br_curr)))

        scale_change = 0.0
        if tl_diff > 1e-6:
            scale_change = (br_diff - tl_diff) / tl_diff * 0.01

        return best_shift_x, best_shift_y, scale_change

    def compute_steadiness_score(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
    ) -> float:
        """Compute how steady the footage is (0-1, 1=perfectly steady).

        Analyzes frame-to-frame jitter to determine camera steadiness.
        Uses frame differencing at center crop to isolate camera motion
        from subject motion.

        Args:
            video_path: Path to the video file.
            start_time: Start time in seconds.
            end_time: End time in seconds.

        Returns:
            Steadiness score from 0 (very shaky) to 1 (perfectly steady).

        Raises:
            FileNotFoundError: If video file doesn't exist.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        width, height = self._get_frame_dimensions(video_path)

        # Sample at 2 FPS for steadiness
        sample_interval = 0.5
        timestamps: list[float] = []
        t = start_time
        while t <= end_time:
            timestamps.append(t)
            t += sample_interval

        if len(timestamps) < 3:
            return 1.0

        # Compute frame-to-frame shifts at center region
        shifts: list[float] = []
        prev_frame: np.ndarray | None = None

        # Use a center crop to reduce influence of subject motion
        crop_margin_x = width // 4
        crop_margin_y = height // 4

        for ts in timestamps:
            frame = self._extract_frame_raw(video_path, ts, width, height)
            if frame is None:
                continue

            # Crop to center
            center_frame = frame[
                crop_margin_y:height - crop_margin_y,
                crop_margin_x:width - crop_margin_x,
            ]

            if prev_frame is not None and center_frame.shape == prev_frame.shape:
                diff = np.abs(center_frame - prev_frame)
                shifts.append(float(np.mean(diff)))

            prev_frame = center_frame.copy()

        if len(shifts) < 2:
            return 1.0

        # Steadiness is inversely related to variation in shifts
        avg_shift = np.mean(shifts)
        std_shift = np.std(shifts)

        # High average shift = lots of motion, high variation = shaky
        # Perfectly steady: low average, low variation
        max_shift = 20.0  # Empirical maximum for "steady"
        shift_score = max(0.0, 1.0 - avg_shift / max_shift)
        variation_score = max(0.0, 1.0 - std_shift / (max_shift * 0.5))

        steadiness = 0.6 * shift_score + 0.4 * variation_score

        return round(max(0.0, min(1.0, steadiness)), 3)
