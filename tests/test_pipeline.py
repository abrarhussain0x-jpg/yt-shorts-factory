"""
tests/test_pipeline.py — Tests for the pipeline orchestrator.
Uses mocks for all core modules to test the pipeline flow, result
structure, and skip behaviour without real downloads or processing.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline import run_pipeline, PipelineResult, StepResult
from core.analyzer import SegmentResult, MultiClipResult
from core.platform_exporter import PlatformExports, ExportResult
from core.metadata_generator import MetadataResult
from core.transcriber import TranscriptionResult
from config.settings import get_settings


def _make_platform_exports(fake_video: Path) -> PlatformExports:
    """Create a PlatformExports with mock ExportResult for each platform."""
    results = []
    for platform_name in ("youtube", "tiktok", "reels"):
        results.append(ExportResult(
            platform=platform_name,
            path=fake_video,
            file_size=1000,
            duration=60.0,
            resolution=(1080, 1920),
            codec="h264",
            crf=23,
            validated=True,
            validation_errors=[],
            variant="default",
            resolution_label="1080x1920",
            codec_label="H.264",
        ))
    return PlatformExports(results=results)


def _make_mock_analyzer(segment: SegmentResult) -> MagicMock:
    """Create a properly mocked EngagementAnalyzer with both analyze methods."""
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = segment
    mock_analyzer.analyze_multiple_clips.return_value = MultiClipResult(
        segments=[segment],
        total_candidates=1,
        analysis_method="multi_signal_v3",
    )
    return mock_analyzer


class TestPipelineResultStructure:
    """Tests for the PipelineResult data class structure."""

    def test_pipeline_result_has_required_fields(self) -> None:
        """PipelineResult should have success, job_id, outputs, metadata, steps, total_duration."""
        result = PipelineResult()
        assert hasattr(result, "success")
        assert hasattr(result, "job_id")
        assert hasattr(result, "outputs")
        assert hasattr(result, "metadata")
        assert hasattr(result, "steps")
        assert hasattr(result, "total_duration_seconds")

    def test_pipeline_result_default_values(self) -> None:
        """Default values should be safe defaults."""
        result = PipelineResult()
        assert result.success is False
        assert result.job_id == ""
        assert result.outputs is None
        assert result.metadata is None
        assert result.steps == []
        assert result.total_duration_seconds == 0.0

    def test_step_result_has_required_fields(self) -> None:
        """StepResult should have name, status, duration, output_path, error."""
        step = StepResult(name="Download", status="done", duration=5.0, output_path="/tmp/test.mp4")
        assert step.name == "Download"
        assert step.status == "done"
        assert step.duration == 5.0
        assert step.output_path == "/tmp/test.mp4"
        assert step.error == ""


class TestPipelineSkipsSubsWhenFlagSet:
    """Tests that the pipeline correctly skips transcription and subtitles when flags are set."""

    @patch("core.pipeline.FaceTracker", create=True)
    @patch("core.pipeline.MotionDetector", create=True)
    @patch("core.pipeline.AudioEnhancer", create=True)
    @patch("core.pipeline.ContentModerator", create=True)
    @patch("core.pipeline.ThumbnailGenerator", create=True)
    @patch("core.pipeline.save_video_record")
    @patch("core.pipeline.update_job_status")
    @patch("core.pipeline.create_job")
    @patch("core.pipeline.init_db")
    @patch("core.pipeline.export_for_platforms")
    @patch("core.pipeline.stamp_logo")
    @patch("core.pipeline.transcribe")
    @patch("core.pipeline.convert_to_shorts")
    @patch("core.pipeline.EngagementAnalyzer")
    @patch("core.pipeline.download_video")
    @patch("core.pipeline.generate_metadata")
    def test_skip_subs_skips_transcribe(
        self,
        mock_metadata,
        mock_download,
        mock_analyzer_cls,
        mock_convert,
        mock_transcribe,
        mock_logo,
        mock_export,
        mock_init_db,
        mock_create_job,
        mock_update_job,
        mock_save_video,
        mock_thumb,
        mock_mod,
        mock_audio,
        mock_motion,
        mock_face,
    ) -> None:
        """When skip_subs=True, the transcribe function should not be called."""
        mock_job = MagicMock()
        mock_job.id = "test-job-123"
        mock_create_job.return_value = mock_job

        tmp = Path(tempfile.mkdtemp())
        fake_video = tmp / "test.mp4"
        fake_video.write_text("fake")

        mock_download.return_value = (
            fake_video,
            {"title": "Test", "id": "abc", "duration": 120.0,
             "uploader": "Test", "view_count": 100, "tags": []},
        )

        segment = SegmentResult(start_time=10.0, end_time=70.0, energy_score=0.8)
        mock_analyzer_cls.return_value = _make_mock_analyzer(segment)

        mock_convert.return_value = fake_video
        mock_logo.return_value = fake_video
        mock_export.return_value = _make_platform_exports(fake_video)
        mock_metadata.return_value = MetadataResult(youtube_title="Test #Shorts")

        result = run_pipeline(
            url="https://www.youtube.com/watch?v=test123",
            skip_subs=True,
            no_logo=True,
        )

        assert isinstance(result, PipelineResult)
        mock_transcribe.assert_not_called()

        transcribe_steps = [s for s in result.steps if s.name == "Transcribe"]
        assert len(transcribe_steps) == 1
        assert transcribe_steps[0].status == "skipped"

        subtitle_steps = [s for s in result.steps if s.name == "Burn Subtitles"]
        assert len(subtitle_steps) == 1
        assert subtitle_steps[0].status == "skipped"

    @patch("core.pipeline.FaceTracker", create=True)
    @patch("core.pipeline.MotionDetector", create=True)
    @patch("core.pipeline.AudioEnhancer", create=True)
    @patch("core.pipeline.ContentModerator", create=True)
    @patch("core.pipeline.ThumbnailGenerator", create=True)
    @patch("core.pipeline.save_video_record")
    @patch("core.pipeline.update_job_status")
    @patch("core.pipeline.create_job")
    @patch("core.pipeline.init_db")
    @patch("core.pipeline.export_for_platforms")
    @patch("core.pipeline.stamp_logo")
    @patch("core.pipeline.transcribe")
    @patch("core.pipeline.convert_to_shorts")
    @patch("core.pipeline.EngagementAnalyzer")
    @patch("core.pipeline.download_video")
    @patch("core.pipeline.generate_metadata")
    def test_no_logo_skips_logo_step(
        self,
        mock_metadata,
        mock_download,
        mock_analyzer_cls,
        mock_convert,
        mock_transcribe,
        mock_logo,
        mock_export,
        mock_init_db,
        mock_create_job,
        mock_update_job,
        mock_save_video,
        mock_thumb,
        mock_mod,
        mock_audio,
        mock_motion,
        mock_face,
    ) -> None:
        """When no_logo=True, the stamp_logo function should not be called."""
        mock_job = MagicMock()
        mock_job.id = "test-job-456"
        mock_create_job.return_value = mock_job

        tmp = Path(tempfile.mkdtemp())
        fake_video = tmp / "test.mp4"
        fake_video.write_text("fake")

        mock_download.return_value = (
            fake_video,
            {"title": "Test", "id": "abc", "duration": 120.0,
             "uploader": "Test", "view_count": 100, "tags": []},
        )

        segment = SegmentResult(start_time=10.0, end_time=70.0, energy_score=0.8)
        mock_analyzer_cls.return_value = _make_mock_analyzer(segment)

        mock_convert.return_value = fake_video
        mock_transcribe.return_value = TranscriptionResult(
            words=[], segments=[], language="en", duration=0.0,
        )
        mock_export.return_value = _make_platform_exports(fake_video)
        mock_metadata.return_value = MetadataResult(youtube_title="Test #Shorts")

        result = run_pipeline(
            url="https://www.youtube.com/watch?v=test123",
            skip_subs=True,
            no_logo=True,
        )

        mock_logo.assert_not_called()

        logo_steps = [s for s in result.steps if s.name == "Stamp Logo"]
        assert len(logo_steps) == 1
        assert logo_steps[0].status == "skipped"

    @patch("core.pipeline.save_video_record")
    @patch("core.pipeline.update_job_status")
    @patch("core.pipeline.create_job")
    @patch("core.pipeline.init_db")
    @patch("core.pipeline.export_for_platforms")
    @patch("core.pipeline.stamp_logo")
    @patch("core.pipeline.generate_subtitles")
    @patch("core.pipeline.burn_subtitles")
    @patch("core.pipeline.transcribe")
    @patch("core.pipeline.convert_to_shorts")
    @patch("core.pipeline.EngagementAnalyzer")
    @patch("core.pipeline.download_video")
    @patch("core.pipeline.generate_metadata")
    def test_download_failure_aborts_pipeline(
        self,
        mock_metadata,
        mock_download,
        mock_analyzer_cls,
        mock_convert,
        mock_transcribe,
        mock_burn,
        mock_gen_subs,
        mock_logo,
        mock_export,
        mock_init_db,
        mock_create_job,
        mock_update_job,
        mock_save_video,
    ) -> None:
        """If download fails, the pipeline should abort and return failed result."""
        from core.downloader import DownloadError

        mock_job = MagicMock()
        mock_job.id = "test-job-789"
        mock_create_job.return_value = mock_job

        mock_download.side_effect = DownloadError("Video not found")

        result = run_pipeline(
            url="https://www.youtube.com/watch?v=badvideo",
        )

        assert result.success is False
        mock_convert.assert_not_called()
        mock_transcribe.assert_not_called()
        mock_export.assert_not_called()
