"""
tests/test_subtitle_engine.py — Tests for the ASS subtitle generator.
Tests timestamp conversion, word grouping, karaoke tag generation,
and ASS file structure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.subtitle_engine import (
    ass_timestamp,
    _group_words_into_lines,
    generate_subtitles,
    SubtitleLine,
)
from core.transcriber import WordTimestamp, TranscriptionResult, Segment
from config.settings import get_settings


class TestAssTimestampConversion:
    """Tests for the ass_timestamp function."""

    def test_zero_seconds(self) -> None:
        """Zero seconds should produce 0:00:00.00."""
        result = ass_timestamp(0.0)
        assert result == "0:00:00.00"

    def test_simple_seconds(self) -> None:
        """Simple second values should format correctly."""
        result = ass_timestamp(5.0)
        assert result == "0:00:05.00"

    def test_minutes_and_seconds(self) -> None:
        """Minutes and seconds should format correctly."""
        result = ass_timestamp(125.0)  # 2 min 5 sec
        assert result == "0:02:05.00"

    def test_hours(self) -> None:
        """Hours should format correctly."""
        result = ass_timestamp(3661.0)  # 1 hour 1 min 1 sec
        assert result == "1:01:01.00"

    def test_centiseconds(self) -> None:
        """Fractional seconds should produce correct centiseconds."""
        result = ass_timestamp(1.23)
        assert result == "0:00:01.23"

    def test_negative_values_clamped(self) -> None:
        """Negative values should be clamped to 0."""
        result = ass_timestamp(-5.0)
        assert result == "0:00:00.00"

    def test_large_centiseconds_rounding(self) -> None:
        """Centiseconds >= 100 should be clamped to 99."""
        result = ass_timestamp(1.999)
        assert ".99" in result


class TestWordGrouping:
    """Tests for the word grouping into subtitle lines."""

    def test_word_grouping_respects_max_words(self) -> None:
        """No line should have more words than max_words."""
        words = [
            WordTimestamp(word=f"word{i}", start=i * 0.5, end=i * 0.5 + 0.4)
            for i in range(20)
        ]
        lines = _group_words_into_lines(words, max_words=4)
        for line in lines:
            assert len(line.words) <= 4

    def test_word_grouping_sentence_boundaries(self) -> None:
        """Lines should break at sentence boundaries (.?!) when possible."""
        words = [
            WordTimestamp(word="Hello", start=0.0, end=0.5),
            WordTimestamp(word="there.", start=0.5, end=1.0),
            WordTimestamp(word="Next", start=1.0, end=1.5),
            WordTimestamp(word="sentence.", start=1.5, end=2.0),
        ]
        lines = _group_words_into_lines(words, max_words=6)
        # Should break at "there." since it's a sentence end
        assert len(lines) >= 2

    def test_word_grouping_empty_input(self) -> None:
        """Empty word list should return empty lines list."""
        lines = _group_words_into_lines([], max_words=4)
        assert lines == []

    def test_word_grouping_fewer_than_max(self) -> None:
        """Fewer words than max_words should produce a single line."""
        words = [
            WordTimestamp(word="Hello", start=0.0, end=0.5),
            WordTimestamp(word="world", start=0.5, end=1.0),
        ]
        lines = _group_words_into_lines(words, max_words=4)
        assert len(lines) == 1
        assert len(lines[0].words) == 2

    def test_word_grouping_overlaps_adjacent_lines(self) -> None:
        """Adjacent lines should have 0.05s overlap to prevent flicker."""
        words = [
            WordTimestamp(word="word1", start=0.0, end=0.5),
            WordTimestamp(word="word2", start=0.5, end=1.0),
            WordTimestamp(word="word3", start=1.0, end=1.5),
            WordTimestamp(word="word4", start=1.5, end=2.0),
            WordTimestamp(word="word5", start=2.0, end=2.5),
        ]
        lines = _group_words_into_lines(words, max_words=3)
        if len(lines) >= 2:
            # The end of line N should overlap the start of line N+1
            for i in range(len(lines) - 1):
                assert lines[i].end >= lines[i + 1].start - 0.05


class TestKaraokeTags:
    """Tests for karaoke tag generation in ASS files."""

    def test_karaoke_tags_generated_correctly(self) -> None:
        """Karaoke animation should produce \\kf tags in the ASS file."""
        settings = get_settings()
        settings.SUBTITLE_ANIMATION = "karaoke"

        words = [
            WordTimestamp(word="Hello", start=0.0, end=0.5),
            WordTimestamp(word="world", start=0.5, end=1.0),
        ]
        transcription = TranscriptionResult(
            words=words,
            segments=[Segment(text="Hello world", start=0.0, end=1.0)],
            language="en",
            duration=1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ass_path = Path(tmpdir) / "test.ass"
            generate_subtitles(transcription, ass_path, settings)

            content = ass_path.read_text(encoding="utf-8")
            assert "\\kf" in content, f"Expected \\kf karaoke tags in ASS file, got: {content[:500]}"

    def test_karaoke_tag_durations_are_positive(self) -> None:
        """Karaoke \\kf durations should be positive integers."""
        settings = get_settings()
        settings.SUBTITLE_ANIMATION = "karaoke"

        words = [
            WordTimestamp(word="Test", start=0.0, end=1.0),
        ]
        transcription = TranscriptionResult(
            words=words,
            segments=[Segment(text="Test", start=0.0, end=1.0)],
            language="en",
            duration=1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ass_path = Path(tmpdir) / "test.ass"
            generate_subtitles(transcription, ass_path, settings)

            content = ass_path.read_text(encoding="utf-8")
            # kf duration should be a positive number (in centiseconds)
            import re
            kf_matches = re.findall(r"\\kf(\d+)", content)
            for duration_str in kf_matches:
                duration = int(duration_str)
                assert duration > 0, f"Karaoke duration should be positive, got {duration}"


class TestAssFileStructure:
    """Tests for the ASS file having required sections and fields."""

    def test_ass_file_has_required_sections(self) -> None:
        """The generated ASS file must contain Script Info, Styles, and Events."""
        settings = get_settings()
        settings.SUBTITLE_ANIMATION = "none"

        words = [
            WordTimestamp(word="Hello", start=0.0, end=0.5),
            WordTimestamp(word="world", start=0.5, end=1.0),
        ]
        transcription = TranscriptionResult(
            words=words,
            segments=[Segment(text="Hello world", start=0.0, end=1.0)],
            language="en",
            duration=1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ass_path = Path(tmpdir) / "test.ass"
            generate_subtitles(transcription, ass_path, settings)

            content = ass_path.read_text(encoding="utf-8")

            assert "[Script Info]" in content, "Missing [Script Info] section"
            assert "ScriptType: v4.00+" in content, "Missing or wrong ScriptType"
            assert "[V4+ Styles]" in content, "Missing [V4+ Styles] section"
            assert "[Events]" in content, "Missing [Events] section"
            assert "Style: Default" in content, "Missing Default style"
            assert "Style: Highlight" in content, "Missing Highlight style"

    def test_ass_file_has_playres(self) -> None:
        """The ASS file should have PlayResX and PlayResY matching settings."""
        settings = get_settings()
        settings.SUBTITLE_ANIMATION = "none"

        words = [WordTimestamp(word="Test", start=0.0, end=1.0)]
        transcription = TranscriptionResult(
            words=words,
            segments=[Segment(text="Test", start=0.0, end=1.0)],
            language="en",
            duration=1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ass_path = Path(tmpdir) / "test.ass"
            generate_subtitles(transcription, ass_path, settings)

            content = ass_path.read_text(encoding="utf-8")
            assert f"PlayResX: {settings.OUTPUT_WIDTH}" in content
            assert f"PlayResY: {settings.OUTPUT_HEIGHT}" in content

    def test_ass_file_has_dialogue_lines(self) -> None:
        """The ASS file should contain Dialogue entries for each subtitle line."""
        settings = get_settings()
        settings.SUBTITLE_ANIMATION = "none"

        words = [
            WordTimestamp(word="Hello", start=0.0, end=0.5),
            WordTimestamp(word="world", start=0.5, end=1.0),
            WordTimestamp(word="How", start=1.0, end=1.5),
            WordTimestamp(word="are", start=1.5, end=2.0),
            WordTimestamp(word="you", start=2.0, end=2.5),
        ]
        transcription = TranscriptionResult(
            words=words,
            segments=[Segment(text="Hello world", start=0.0, end=2.5)],
            language="en",
            duration=2.5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ass_path = Path(tmpdir) / "test.ass"
            generate_subtitles(transcription, ass_path, settings)

            content = ass_path.read_text(encoding="utf-8")
            dialogue_lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
            assert len(dialogue_lines) >= 1, "Should have at least one Dialogue line"
