"""
core/__init__.py — Package initializer for core module.
"""

from core.downloader import download_video, DownloadError
from core.analyzer import EngagementAnalyzer, SegmentResult, MultiClipResult, visualize_profile
from core.face_tracker import FaceTracker, SmartCropResult, CropPosition, FaceBox, BodyBox, TrackedFace
from core.motion_detector import MotionDetector, OpticalFlowResult, CameraMotionResult, MotionFrame, MovingObject
from core.audio_enhancer import AudioEnhancer, AudioQualityReport
from core.shorts_converter import convert_to_shorts
from core.transcriber import transcribe, TranscriptionResult
from core.subtitle_engine import generate_subtitles, burn_subtitles
from core.logo_stamper import stamp_logo
from core.metadata_generator import generate_metadata
from core.platform_exporter import export_for_platforms
from core.pipeline import run_pipeline

__all__ = [
    # Downloader
    "download_video",
    "DownloadError",
    # Analyzer
    "EngagementAnalyzer",
    "SegmentResult",
    "MultiClipResult",
    "visualize_profile",
    # Face Tracker
    "FaceTracker",
    "SmartCropResult",
    "CropPosition",
    "FaceBox",
    "BodyBox",
    "TrackedFace",
    # Motion Detector
    "MotionDetector",
    "OpticalFlowResult",
    "CameraMotionResult",
    "MotionFrame",
    "MovingObject",
    # Audio Enhancer
    "AudioEnhancer",
    "AudioQualityReport",
    # Other core modules
    "convert_to_shorts",
    "transcribe",
    "TranscriptionResult",
    "generate_subtitles",
    "burn_subtitles",
    "stamp_logo",
    "generate_metadata",
    "export_for_platforms",
    "run_pipeline",
]
