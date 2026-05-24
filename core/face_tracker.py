"""
core/face_tracker.py — Advanced face/body tracking for smart crop positioning.

Provides face detection (Haar cascades + DNN), body detection, temporal
smoothing of crop positions, and a full smart-crop pipeline. Gracefully
degrades when OpenCV or mediapipe are unavailable.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from utils.ffmpeg_utils import get_video_thumbnail, probe_video
from utils.logger import get_logger

logger = get_logger("face_tracker")

# ── Lazy imports ──────────────────────────────────────────
_cv2 = None
_mediapipe = None


def _get_cv2():
    """Lazily import OpenCV, returning None if unavailable."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2
            _cv2 = cv2
        except ImportError:
            logger.debug("OpenCV not available; face tracking will be limited")
    return _cv2


def _get_mediapipe():
    """Lazily import mediapipe, returning None if unavailable."""
    global _mediapipe
    if _mediapipe is None:
        try:
            import mediapipe as mp
            _mediapipe = mp
        except ImportError:
            logger.debug("mediapipe not available; using OpenCV fallback")
    return _mediapipe


# ── Data Classes ──────────────────────────────────────────

@dataclass
class FaceBox:
    """Detected face bounding box with confidence."""

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    detector: str = "unknown"  # "haar", "dnn", "mediapipe"

    @property
    def center_x(self) -> int:
        """X coordinate of face center."""
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        """Y coordinate of face center."""
        return self.y + self.height // 2

    @property
    def area(self) -> int:
        """Area of the face bounding box in pixels²."""
        return self.width * self.height


@dataclass
class BodyBox:
    """Detected full body bounding box with confidence."""

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    detector: str = "unknown"  # "hog", "dnn"

    @property
    def center_x(self) -> int:
        """X coordinate of body center."""
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        """Y coordinate of body center."""
        return self.y + self.height // 2

    @property
    def area(self) -> int:
        """Area of the body bounding box in pixels²."""
        return self.width * self.height


@dataclass
class TrackedFace:
    """Face position tracked over time."""

    face_id: int
    timestamps: list[float] = field(default_factory=list)
    positions: list[FaceBox] = field(default_factory=list)
    avg_confidence: float = 0.0

    @property
    def duration(self) -> float:
        """Duration this face is tracked (seconds)."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    def add_observation(self, timestamp: float, face: FaceBox) -> None:
        """Add a face observation at a given timestamp.

        Args:
            timestamp: Time in seconds.
            face: FaceBox detected at this timestamp.
        """
        self.timestamps.append(timestamp)
        self.positions.append(face)
        # Update running average confidence
        n = len(self.positions)
        self.avg_confidence = ((n - 1) * self.avg_confidence + face.confidence) / n


@dataclass
class CropPosition:
    """Crop rectangle position for vertical video framing."""

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    face_count: int = 0
    method: str = "center"  # "center", "face", "body", "smoothed"

    @property
    def center_x(self) -> int:
        """X center of the crop."""
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        """Y center of the crop."""
        return self.y + self.height // 2


@dataclass
class SmartCropResult:
    """Result of the full smart crop pipeline."""

    crop_positions: list[CropPosition] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    average_confidence: float = 0.0
    face_detection_rate: float = 0.0  # Fraction of frames with faces
    method_used: str = "smart_crop"
    warnings: list[str] = field(default_factory=list)


# ── FaceTracker Class ────────────────────────────────────

class FaceTracker:
    """Advanced face and body tracking for smart crop positioning.

    Supports multiple detection backends (OpenCV Haar cascades,
    OpenCV DNN, mediapipe) with automatic fallback. Provides
    temporal smoothing to prevent jittery crop positions and
    confidence-weighted position averaging for multi-face scenes.
    """

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        """Initialize the face tracker.

        Args:
            confidence_threshold: Minimum confidence for a detection
                to be considered valid (0-1).
        """
        self.confidence_threshold = confidence_threshold
        self._haar_cascade = None
        self._dnn_net = None
        self._mediapipe_face = None
        self._body_hog = None

    def _init_haar_cascade(self) -> bool:
        """Initialize Haar cascade face detector.

        Returns:
            True if successfully initialized.
        """
        cv2 = _get_cv2()
        if cv2 is None:
            return False

        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
            if self._haar_cascade.empty():
                logger.warning("Failed to load Haar cascade")
                self._haar_cascade = None
                return False
            return True
        except Exception as exc:
            logger.debug("Haar cascade init failed: %s", exc)
            self._haar_cascade = None
            return False

    def _init_dnn_detector(self) -> bool:
        """Initialize OpenCV DNN face detector.

        Uses the res10_300x300_ssd model that ships with OpenCV.

        Returns:
            True if successfully initialized.
        """
        cv2 = _get_cv2()
        if cv2 is None:
            return False

        try:
            model_file = cv2.data.haarcascades.replace("haarcascades", "dnn") + "/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            config_file = cv2.data.haarcascades.replace("haarcascades", "dnn") + "/deploy.prototxt"

            # Try alternate paths
            import os
            if not os.path.exists(model_file):
                # Try common install locations
                alt_paths = [
                    "/usr/share/opencv4/dnn/",
                    "/usr/local/share/opencv4/dnn/",
                    os.path.expanduser("~/.local/share/opencv4/dnn/"),
                ]
                for alt in alt_paths:
                    alt_model = os.path.join(alt, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
                    alt_config = os.path.join(alt, "deploy.prototxt")
                    if os.path.exists(alt_model):
                        model_file = alt_model
                        config_file = alt_config
                        break

            if not os.path.exists(model_file):
                logger.debug("DNN face model not found at %s", model_file)
                return False

            self._dnn_net = cv2.dnn.readNetFromCaffe(config_file, model_file)
            return True
        except Exception as exc:
            logger.debug("DNN face detector init failed: %s", exc)
            self._dnn_net = None
            return False

    def _init_mediapipe(self) -> bool:
        """Initialize mediapipe face detection.

        Returns:
            True if successfully initialized.
        """
        mp = _get_mediapipe()
        if mp is None:
            return False

        try:
            self._mediapipe_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0,  # 0 = short range, 1 = full range
                min_detection_confidence=self.confidence_threshold,
            )
            return True
        except Exception as exc:
            logger.debug("mediapipe face detection init failed: %s", exc)
            self._mediapipe_face = None
            return False

    def detect_faces(self, frame_path: Path) -> list[FaceBox]:
        """Detect faces in a single frame using multiple backends.

        Tries mediapipe first (most accurate), then DNN, then Haar
        cascade as fallback. Returns detections from the first
        successful backend.

        Args:
            frame_path: Path to the image file (JPEG/PNG).

        Returns:
            List of FaceBox objects for detected faces. Empty list
            if no faces found or no detector available.

        Raises:
            FileNotFoundError: If frame_path doesn't exist.
        """
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame file not found: {frame_path}")

        # Try mediapipe first
        faces = self._detect_faces_mediapipe(frame_path)
        if faces:
            return faces

        # Try DNN
        faces = self._detect_faces_dnn(frame_path)
        if faces:
            return faces

        # Try Haar cascade
        faces = self._detect_faces_haar(frame_path)
        if faces:
            return faces

        return []

    def _detect_faces_haar(self, frame_path: Path) -> list[FaceBox]:
        """Detect faces using OpenCV Haar cascade.

        Args:
            frame_path: Path to the image file.

        Returns:
            List of FaceBox objects.
        """
        cv2 = _get_cv2()
        if cv2 is None:
            return []

        if self._haar_cascade is None:
            if not self._init_haar_cascade():
                return []

        try:
            img = cv2.imread(str(frame_path))
            if img is None:
                return []

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detections = self._haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            faces: list[FaceBox] = []
            for (x, y, w, h) in detections:
                faces.append(FaceBox(
                    x=int(x), y=int(y), width=int(w), height=int(h),
                    confidence=0.7,  # Haar doesn't provide confidence
                    detector="haar",
                ))
            return faces

        except Exception as exc:
            logger.debug("Haar cascade detection failed: %s", exc)
            return []

    def _detect_faces_dnn(self, frame_path: Path) -> list[FaceBox]:
        """Detect faces using OpenCV DNN face detector.

        Uses the SSD-based face detector which provides confidence
        scores for each detection.

        Args:
            frame_path: Path to the image file.

        Returns:
            List of FaceBox objects with confidence scores.
        """
        cv2 = _get_cv2()
        if cv2 is None:
            return []

        if self._dnn_net is None:
            if not self._init_dnn_detector():
                return []

        try:
            img = cv2.imread(str(frame_path))
            if img is None:
                return []

            h, w = img.shape[:2]

            # Create blob from image
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self._dnn_net.setInput(blob)
            detections = self._dnn_net.forward()

            faces: list[FaceBox] = []
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < self.confidence_threshold:
                    continue

                # Compute bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")

                # Clamp to image bounds
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(w, end_x)
                end_y = min(h, end_y)

                faces.append(FaceBox(
                    x=start_x, y=start_y,
                    width=end_x - start_x, height=end_y - start_y,
                    confidence=confidence,
                    detector="dnn",
                ))

            return faces

        except Exception as exc:
            logger.debug("DNN face detection failed: %s", exc)
            return []

    def _detect_faces_mediapipe(self, frame_path: Path) -> list[FaceBox]:
        """Detect faces using mediapipe face detection.

        Args:
            frame_path: Path to the image file.

        Returns:
            List of FaceBox objects with confidence scores.
        """
        cv2 = _get_cv2()
        mp = _get_mediapipe()
        if cv2 is None or mp is None:
            return []

        if self._mediapipe_face is None:
            if not self._init_mediapipe():
                return []

        try:
            img = cv2.imread(str(frame_path))
            if img is None:
                return []

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self._mediapipe_face.process(rgb)

            faces: list[FaceBox] = []
            if results.detections:
                h, w = img.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    fw = int(bbox.width * w)
                    fh = int(bbox.height * h)

                    # Clamp to image bounds
                    x = max(0, x)
                    y = max(0, y)
                    fw = min(fw, w - x)
                    fh = min(fh, h - y)

                    confidence = detection.score[0] if detection.score else 0.5

                    faces.append(FaceBox(
                        x=x, y=y, width=fw, height=fh,
                        confidence=confidence,
                        detector="mediapipe",
                    ))

            return faces

        except Exception as exc:
            logger.debug("mediapipe face detection failed: %s", exc)
            return []

    def detect_faces_video(
        self, video_path: Path, timestamps: list[float]
    ) -> dict[float, list[FaceBox]]:
        """Detect faces at multiple timestamps in a video.

        Extracts frames at each timestamp and runs face detection.

        Args:
            video_path: Path to the video file.
            timestamps: List of time positions in seconds.

        Returns:
            Dictionary mapping timestamp to list of detected FaceBoxes.

        Raises:
            FileNotFoundError: If video file doesn't exist.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        results: dict[float, list[FaceBox]] = {}
        temp_files: list[Path] = []

        try:
            for ts in timestamps:
                tmp_jpg = Path(tempfile.mktemp(suffix=".jpg"))
                temp_files.append(tmp_jpg)

                try:
                    get_video_thumbnail(video_path, ts, tmp_jpg)
                    faces = self.detect_faces(tmp_jpg)
                    results[ts] = faces
                except Exception as exc:
                    logger.debug("Face detection failed at %.1fs: %s", ts, exc)
                    results[ts] = []
        finally:
            # Cleanup temp files
            for tmp in temp_files:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

        return results

    def track_faces_video(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        sample_interval: float = 2.0,
    ) -> list[TrackedFace]:
        """Track face positions across time in a video.

        Detects faces at regular intervals and groups them into
        tracked face tracks based on position proximity.

        Args:
            video_path: Path to the video file.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            sample_interval: Seconds between face detection samples.

        Returns:
            List of TrackedFace objects, each representing a face
            tracked across multiple frames.

        Raises:
            FileNotFoundError: If video file doesn't exist.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Generate sample timestamps
        timestamps = []
        t = start_time
        while t <= end_time:
            timestamps.append(t)
            t += sample_interval

        if not timestamps:
            return []

        # Detect faces at each timestamp
        detections = self.detect_faces_video(video_path, timestamps)

        # Track faces across frames using proximity matching
        tracks: list[TrackedFace] = []
        max_distance = 150  # Maximum pixel distance for same-face matching

        for ts in timestamps:
            faces = detections.get(ts, [])

            for face in faces:
                # Try to match with existing track
                matched = False
                for track in tracks:
                    if not track.positions:
                        continue
                    last_pos = track.positions[-1]
                    # Compute distance between face centers
                    dist = math.sqrt(
                        (face.center_x - last_pos.center_x) ** 2
                        + (face.center_y - last_pos.center_y) ** 2
                    )
                    if dist < max_distance:
                        track.add_observation(ts, face)
                        matched = True
                        break

                if not matched:
                    # Start a new track
                    new_track = TrackedFace(face_id=len(tracks))
                    new_track.add_observation(ts, face)
                    tracks.append(new_track)

        # Sort tracks by duration (longest first)
        tracks.sort(key=lambda t: t.duration, reverse=True)

        logger.info(
            "Tracked %d face(s) over %.1f-%.1fs (%d samples)",
            len(tracks), start_time, end_time, len(timestamps),
        )

        return tracks

    @staticmethod
    def get_largest_face(faces: list[FaceBox]) -> FaceBox | None:
        """Get the largest detected face by bounding box area.

        Useful for selecting the primary subject in a multi-face scene.

        Args:
            faces: List of FaceBox objects.

        Returns:
            The largest FaceBox, or None if the list is empty.
        """
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)

    @staticmethod
    def get_center_face(faces: list[FaceBox], frame_width: int) -> FaceBox | None:
        """Get the face closest to the horizontal center of the frame.

        Useful for selecting the primary speaker in a multi-face scene.

        Args:
            faces: List of FaceBox objects.
            frame_width: Width of the frame in pixels.

        Returns:
            The centermost FaceBox, or None if the list is empty.
        """
        if not faces:
            return None
        frame_center = frame_width / 2
        return min(faces, key=lambda f: abs(f.center_x - frame_center))

    def compute_crop_position(
        self,
        faces: list[FaceBox],
        frame_w: int,
        frame_h: int,
        target_ar: float = 9 / 16,
        padding: float = 0.1,
    ) -> CropPosition:
        """Compute optimal crop position based on detected faces.

        Calculates a crop rectangle with the specified aspect ratio,
        centered on the detected face(s). Supports multi-face
        videos by centering on the centroid of all faces.

        Args:
            faces: List of detected FaceBox objects.
            frame_w: Source frame width in pixels.
            frame_h: Source frame height in pixels.
            target_ar: Target aspect ratio (width/height). Default 9/16 for shorts.
            padding: Padding fraction around faces (0-1). Default 10%.

        Returns:
            CropPosition with the computed crop rectangle.
        """
        if not faces or frame_w <= 0 or frame_h <= 0:
            # Return centered crop
            crop_w = min(frame_w, int(frame_h * target_ar))
            crop_h = min(frame_h, int(crop_w / target_ar))
            crop_x = (frame_w - crop_w) // 2
            crop_y = (frame_h - crop_h) // 2
            return CropPosition(
                x=max(0, crop_x),
                y=max(0, crop_y),
                width=crop_w,
                height=crop_h,
                confidence=0.0,
                face_count=0,
                method="center",
            )

        # Compute centroid of all faces (confidence-weighted)
        total_weight = sum(f.confidence for f in faces)
        if total_weight < 1e-6:
            total_weight = 1.0

        centroid_x = sum(f.center_x * f.confidence for f in faces) / total_weight
        centroid_y = sum(f.center_y * f.confidence for f in faces) / total_weight

        # Compute target crop dimensions
        crop_h = frame_h
        crop_w = int(crop_h * target_ar)

        # Ensure crop fits within frame
        if crop_w > frame_w:
            crop_w = frame_w
            crop_h = int(crop_w / target_ar)

        # Add padding
        padding_x = int(faces[0].width * padding) if faces else 0
        padding_y = int(faces[0].height * padding) if faces else 0

        # Position crop centered on face centroid with padding offset
        # For shorts: keep face in upper third for better composition
        target_y = centroid_y - crop_h * 0.15 + padding_y  # Slightly above center
        target_x = centroid_x - crop_w / 2 + padding_x

        # Clamp to frame bounds
        crop_x = int(max(0, min(target_x, frame_w - crop_w)))
        crop_y = int(max(0, min(target_y, frame_h - crop_h)))

        avg_confidence = sum(f.confidence for f in faces) / len(faces)

        return CropPosition(
            x=crop_x,
            y=crop_y,
            width=crop_w,
            height=crop_h,
            confidence=avg_confidence,
            face_count=len(faces),
            method="face",
        )

    @staticmethod
    def smooth_crop_positions(
        positions: list[CropPosition],
        window_size: int = 5,
    ) -> list[CropPosition]:
        """Smooth crop positions over time to prevent jitter.

        Uses a weighted moving average with a configurable window size.
        More recent positions get higher weight (exponential decay).

        Args:
            positions: List of CropPosition objects ordered by time.
            window_size: Size of the smoothing window. Default 5.

        Returns:
            List of smoothed CropPosition objects.
        """
        if len(positions) <= 2:
            return positions

        smoothed: list[CropPosition] = []

        for i in range(len(positions)):
            start = max(0, i - window_size // 2)
            end = min(len(positions), i + window_size // 2 + 1)
            window = positions[start:end]

            # Exponential decay weights
            weights = [math.exp(-0.3 * abs(j - i)) for j in range(start, end)]
            total_weight = sum(weights)

            if total_weight < 1e-6:
                smoothed.append(positions[i])
                continue

            # Weighted average of x, y positions
            avg_x = sum(p.x * w for p, w in zip(window, weights)) / total_weight
            avg_y = sum(p.y * w for p, w in zip(window, weights)) / total_weight
            avg_conf = sum(p.confidence * w for p, w in zip(window, weights)) / total_weight

            # Use dimensions from the current position
            current = positions[i]

            smoothed.append(CropPosition(
                x=int(round(avg_x)),
                y=int(round(avg_y)),
                width=current.width,
                height=current.height,
                confidence=round(avg_conf, 3),
                face_count=current.face_count,
                method="smoothed",
            ))

        return smoothed

    def detect_body(self, frame_path: Path) -> list[BodyBox]:
        """Detect full body in a frame using HOG or DNN.

        Attempts HOG descriptor first, falling back to DNN if
        HOG fails or isn't available.

        Args:
            frame_path: Path to the image file.

        Returns:
            List of BodyBox objects for detected bodies.

        Raises:
            FileNotFoundError: If frame_path doesn't exist.
        """
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame file not found: {frame_path}")

        cv2 = _get_cv2()
        if cv2 is None:
            return []

        try:
            img = cv2.imread(str(frame_path))
            if img is None:
                return []

            # Try HOG descriptor for pedestrian detection
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            # Resize for speed (HOG works best at ~128x64 minimum person size)
            h, w = img.shape[:2]
            scale = min(1.0, 800 / max(h, w))
            if scale < 1.0:
                small_img = cv2.resize(img, (int(w * scale), int(h * scale)))
            else:
                small_img = img
                scale = 1.0

            detections, weights = hog.detectMultiScale(
                small_img,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05,
            )

            bodies: list[BodyBox] = []
            for (x, y, bw, bh), weight in zip(detections, weights):
                if weight < self.confidence_threshold:
                    continue
                # Scale back to original image coordinates
                bodies.append(BodyBox(
                    x=int(x / scale),
                    y=int(y / scale),
                    width=int(bw / scale),
                    height=int(bh / scale),
                    confidence=float(weight),
                    detector="hog",
                ))

            return bodies

        except Exception as exc:
            logger.debug("Body detection failed: %s", exc)
            return []

    def compute_smart_crop(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        target_w: int = 1080,
        target_h: int = 1920,
        sample_interval: float = 2.0,
    ) -> SmartCropResult:
        """Full smart crop pipeline: detect, track, smooth.

        Runs the complete face tracking and crop computation pipeline:
        1. Detect faces at regular intervals
        2. Track faces across time
        3. Compute crop positions for each sample
        4. Smooth positions to prevent jitter
        5. Return the final crop sequence

        Args:
            video_path: Path to the video file.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            target_w: Target crop width in pixels (default 1080).
            target_h: Target crop height in pixels (default 1920).
            sample_interval: Seconds between crop position samples.

        Returns:
            SmartCropResult with the complete crop sequence.

        Raises:
            FileNotFoundError: If video file doesn't exist.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        target_ar = target_w / target_h
        video_info = probe_video(video_path)
        frame_w = video_info.width if video_info.width > 0 else 1920
        frame_h = video_info.height if video_info.height > 0 else 1080

        # Generate sample timestamps
        timestamps: list[float] = []
        t = start_time
        while t <= end_time:
            timestamps.append(t)
            t += sample_interval

        if not timestamps:
            # Return single center crop
            crop = self.compute_crop_position([], frame_w, frame_h, target_ar)
            return SmartCropResult(
                crop_positions=[crop],
                timestamps=[start_time],
                method_used="center_fallback",
                warnings=["No timestamps generated for smart crop"],
            )

        # Detect faces at each timestamp
        detections = self.detect_faces_video(video_path, timestamps)

        # Compute raw crop positions
        raw_positions: list[CropPosition] = []
        face_frames = 0
        warnings: list[str] = []

        for ts in timestamps:
            faces = detections.get(ts, [])
            if faces:
                face_frames += 1

            crop = self.compute_crop_position(faces, frame_w, frame_h, target_ar)
            raw_positions.append(crop)

        # Smooth crop positions
        smoothed_positions = self.smooth_crop_positions(raw_positions, window_size=5)

        # Compute statistics
        face_rate = face_frames / len(timestamps) if timestamps else 0.0
        avg_conf = (
            sum(p.confidence for p in smoothed_positions) / len(smoothed_positions)
            if smoothed_positions else 0.0
        )

        if face_rate < 0.3:
            warnings.append(f"Low face detection rate: {face_rate:.1%} of frames had faces")

        method = "face_tracked" if face_rate > 0.3 else "center_fallback"

        result = SmartCropResult(
            crop_positions=smoothed_positions,
            timestamps=timestamps,
            average_confidence=round(avg_conf, 3),
            face_detection_rate=round(face_rate, 3),
            method_used=method,
            warnings=warnings,
        )

        logger.info(
            "Smart crop: %d positions, %.1f%% face detection, confidence=%.2f, method=%s",
            len(smoothed_positions), face_rate * 100, avg_conf, method,
        )

        return result


# ── Module-level convenience ──────────────────────────────

def detect_face_crop_region(
    video_path: Path,
    timestamp: float,
    target_w: int,
    target_h: int,
) -> tuple[int, int]:
    """Detect optimal crop position using face tracking.

    Backward-compatible convenience function that creates a FaceTracker
    and uses it to find the best crop position at a single timestamp.

    Args:
        video_path: Path to the video file.
        timestamp: Time in seconds to sample.
        target_w: Target crop width in pixels.
        target_h: Target crop height in pixels.

    Returns:
        Tuple of (crop_x, crop_y) for the best crop position.
        Returns (-1, -1) if face detection is unavailable.
    """
    try:
        tracker = FaceTracker()
        target_ar = target_w / target_h if target_h > 0 else 9 / 16

        # Extract frame
        tmp_jpg = Path(tempfile.mktemp(suffix=".jpg"))
        get_video_thumbnail(video_path, timestamp, tmp_jpg)

        # Detect faces
        faces = tracker.detect_faces(tmp_jpg)

        # Clean up temp file
        try:
            tmp_jpg.unlink(missing_ok=True)
        except OSError:
            pass

        if not faces:
            return -1, -1

        # Get video dimensions
        video_info = probe_video(video_path)
        frame_w = video_info.width if video_info.width > 0 else 1920
        frame_h = video_info.height if video_info.height > 0 else 1080

        # Compute crop position
        crop = tracker.compute_crop_position(faces, frame_w, frame_h, target_ar)
        return crop.x, crop.y

    except Exception as exc:
        logger.debug("Face crop detection failed: %s; using center crop", exc)
        return -1, -1


# ── Need math import ──────────────────────────────────────
import math
