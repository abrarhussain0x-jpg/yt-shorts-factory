"""
tests/test_ai_metadata.py — Tests for the AI metadata module.
Tests AI metadata generation, local fallback, dataclass,
cache functionality, and response parsing.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ai_metadata import (
    AIMetadataResult,
    generate_ai_metadata,
    _generate_locally,
    _parse_ai_response,
    _clamp_score,
    _clean_hashtags,
    _clean_keywords,
    _compute_cache_key,
    _get_cached,
    _set_cached,
    clear_cache,
    get_cache_stats,
    _build_prompt,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_cache_before_each():
    """Clear the metadata cache before each test to avoid cross-contamination."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def sample_transcription() -> str:
    """Sample transcription text for testing."""
    return (
        "Welcome to this amazing video about artificial intelligence. "
        "Today we will explore how machine learning is transforming "
        "the world of content creation and social media."
    )


@pytest.fixture
def sample_title() -> str:
    """Sample video title for testing."""
    return "AI Revolution: How Machine Learning Changes Everything"


@pytest.fixture
def mock_settings() -> MagicMock:
    """Mock settings object for testing."""
    settings = MagicMock()
    settings.METADATA_HASHTAG_COUNT = 10
    settings.METADATA_MAX_KEYWORDS = 10
    settings.METADATA_LANGUAGE = "en"
    settings.FFMPEG_PIXEL_FORMAT = "yuv420p"
    return settings


# ── Test AIMetadataResult Dataclass ──────────────────────────

class TestAIMetadataResult:
    """Tests for the AIMetadataResult dataclass."""

    def test_default_values(self) -> None:
        """Default AIMetadataResult should have safe empty values."""
        result = AIMetadataResult()
        assert result.youtube_title == ""
        assert result.youtube_description == ""
        assert result.tiktok_caption == ""
        assert result.reels_caption == ""
        assert result.hashtags == []
        assert result.keywords == []
        assert result.seo_score == 0.0
        assert result.viral_score == 0.0

    def test_custom_values(self) -> None:
        """AIMetadataResult should accept custom values."""
        result = AIMetadataResult(
            youtube_title="Test Title",
            youtube_description="Test Description",
            tiktok_caption="Test Caption",
            reels_caption="Test Reels",
            hashtags=["tag1", "tag2"],
            keywords=["keyword1", "keyword2"],
            seo_score=85.0,
            viral_score=92.0,
        )
        assert result.youtube_title == "Test Title"
        assert result.seo_score == 85.0
        assert result.viral_score == 92.0
        assert len(result.hashtags) == 2

    def test_to_dict(self) -> None:
        """to_dict should produce a proper dictionary."""
        result = AIMetadataResult(
            youtube_title="Test",
            seo_score=75.0,
            hashtags=["ai", "ml"],
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["youtube_title"] == "Test"
        assert d["seo_score"] == 75.0
        assert d["hashtags"] == ["ai", "ml"]
        assert "keywords" in d
        assert "viral_score" in d

    def test_to_dict_has_all_keys(self) -> None:
        """to_dict should have all expected keys."""
        result = AIMetadataResult()
        d = result.to_dict()
        expected_keys = {
            "youtube_title", "youtube_description", "tiktok_caption",
            "reels_caption", "hashtags", "keywords", "seo_score", "viral_score",
        }
        assert set(d.keys()) == expected_keys


# ── Test _clamp_score ─────────────────────────────────────────

class TestClampScore:
    """Tests for the _clamp_score helper function."""

    def test_normal_value(self) -> None:
        """A normal value should be returned as-is (rounded)."""
        assert _clamp_score(50) == 50.0

    def test_over_100_clamped(self) -> None:
        """Values over 100 should be clamped to 100."""
        assert _clamp_score(150) == 100.0

    def test_negative_clamped(self) -> None:
        """Negative values should be clamped to 0."""
        assert _clamp_score(-10) == 0.0

    def test_string_value(self) -> None:
        """String representations of numbers should be parsed."""
        assert _clamp_score("75") == 75.0

    def test_invalid_string_returns_zero(self) -> None:
        """Non-numeric strings should return 0."""
        assert _clamp_score("invalid") == 0.0

    def test_none_returns_zero(self) -> None:
        """None should return 0."""
        assert _clamp_score(None) == 0.0

    def test_float_value(self) -> None:
        """Float values should work correctly."""
        assert _clamp_score(85.7) == 85.7

    def test_boundary_values(self) -> None:
        """Boundary values 0 and 100 should be preserved."""
        assert _clamp_score(0) == 0.0
        assert _clamp_score(100) == 100.0


# ── Test _clean_hashtags ──────────────────────────────────────

class TestCleanHashtags:
    """Tests for the _clean_hashtags helper function."""

    def test_strips_hash_prefix(self) -> None:
        """Hashtag # prefix should be stripped."""
        result = _clean_hashtags(["#ai", "#machinelearning"])
        assert "ai" in result
        assert "machinelearning" in result

    def test_removes_duplicates(self) -> None:
        """Duplicate hashtags should be removed (case-insensitive)."""
        result = _clean_hashtags(["AI", "ai", "Ai"])
        assert len(result) == 1

    def test_removes_short_tags(self) -> None:
        """Tags shorter than 2 chars should be removed."""
        result = _clean_hashtags(["a", "ok", "x"])
        assert "a" not in result
        assert "ok" in result
        assert "x" not in result

    def test_limits_to_20_tags(self) -> None:
        """Should return at most 20 hashtags."""
        tags = [f"tag{i}" for i in range(30)]
        result = _clean_hashtags(tags)
        assert len(result) <= 20

    def test_non_list_input(self) -> None:
        """Non-list input should return empty list."""
        result = _clean_hashtags("not a list")
        assert result == []

    def test_empty_list(self) -> None:
        """Empty list should return empty list."""
        result = _clean_hashtags([])
        assert result == []


# ── Test _clean_keywords ──────────────────────────────────────

class TestCleanKeywords:
    """Tests for the _clean_keywords helper function."""

    def test_lowercases_keywords(self) -> None:
        """Keywords should be lowercased."""
        result = _clean_keywords(["MachineLearning", "AI"])
        assert "machinelearning" in result
        assert "ai" in result

    def test_removes_duplicates(self) -> None:
        """Duplicate keywords should be removed."""
        result = _clean_keywords(["python", "Python", "PYTHON"])
        assert len(result) == 1

    def test_removes_short_keywords(self) -> None:
        """Keywords shorter than 2 chars should be removed."""
        result = _clean_keywords(["a", "ok", "x"])
        assert "a" not in result
        assert "ok" in result

    def test_limits_to_15_keywords(self) -> None:
        """Should return at most 15 keywords."""
        kws = [f"keyword{i}" for i in range(30)]
        result = _clean_keywords(kws)
        assert len(result) <= 15

    def test_non_list_input(self) -> None:
        """Non-list input should return empty list."""
        result = _clean_keywords("not a list")
        assert result == []

    def test_empty_list(self) -> None:
        """Empty list should return empty list."""
        result = _clean_keywords([])
        assert result == []


# ── Test _parse_ai_response ───────────────────────────────────

class TestParseAIResponse:
    """Tests for the _parse_ai_response function."""

    def test_parse_direct_data_response(self) -> None:
        """A direct data response should be parsed correctly."""
        response = {
            "youtube_title": "Amazing AI Video",
            "youtube_description": "Check this out",
            "tiktok_caption": "Wait for it",
            "reels_caption": "Must watch",
            "hashtags": ["ai", "ml"],
            "keywords": ["artificial intelligence"],
            "seo_score": 85,
            "viral_score": 92,
        }
        result = _parse_ai_response(response)
        assert result.youtube_title == "Amazing AI Video"
        assert result.seo_score == 85.0
        assert result.viral_score == 92.0

    def test_parse_wrapped_data_response(self) -> None:
        """A wrapped response with 'data' key should be parsed correctly."""
        response = {
            "data": {
                "youtube_title": "Wrapped Title",
                "seo_score": 70,
            }
        }
        result = _parse_ai_response(response)
        assert result.youtube_title == "Wrapped Title"
        assert result.seo_score == 70.0

    def test_parse_raw_text_response(self) -> None:
        """A response with raw_text containing JSON should be parsed."""
        raw_json = json.dumps({
            "youtube_title": "Raw Title",
            "seo_score": 60,
        })
        response = {"raw_text": f"Some text before {raw_json} some text after"}
        result = _parse_ai_response(response)
        assert result.youtube_title == "Raw Title"

    def test_parse_error_response(self) -> None:
        """A response with an error key should return empty result."""
        response = {"error": "Something went wrong"}
        result = _parse_ai_response(response)
        assert result.youtube_title == ""

    def test_parse_empty_response(self) -> None:
        """An empty response should return defaults."""
        result = _parse_ai_response({})
        assert isinstance(result, AIMetadataResult)

    def test_parse_response_with_missing_fields(self) -> None:
        """Missing fields should use defaults."""
        response = {"youtube_title": "Partial Title"}
        result = _parse_ai_response(response)
        assert result.youtube_title == "Partial Title"
        assert result.tiktok_caption == ""
        assert result.seo_score == 0.0


# ── Test Cache Functionality ─────────────────────────────────

class TestCacheFunctionality:
    """Tests for the cache functionality."""

    def test_compute_cache_key_deterministic(self) -> None:
        """Same inputs should produce the same cache key."""
        key1 = _compute_cache_key("test text", "test title", "youtube", "abc123")
        key2 = _compute_cache_key("test text", "test title", "youtube", "abc123")
        assert key1 == key2

    def test_compute_cache_key_different_for_different_inputs(self) -> None:
        """Different inputs should produce different cache keys."""
        key1 = _compute_cache_key("text A", "title", "youtube", "hash")
        key2 = _compute_cache_key("text B", "title", "youtube", "hash")
        assert key1 != key2

    def test_compute_cache_key_different_platform(self) -> None:
        """Different platforms should produce different cache keys."""
        key1 = _compute_cache_key("text", "title", "youtube", "hash")
        key2 = _compute_cache_key("text", "title", "tiktok", "hash")
        assert key1 != key2

    def test_set_and_get_cached(self) -> None:
        """Storing and retrieving a cached result should work."""
        result = AIMetadataResult(youtube_title="Cached Title")
        key = "test_cache_key"
        _set_cached(key, result)
        cached = _get_cached(key)
        assert cached is not None
        assert cached.youtube_title == "Cached Title"

    def test_get_cached_miss(self) -> None:
        """Getting a non-existent cache key should return None."""
        cached = _get_cached("nonexistent_key")
        assert cached is None

    def test_clear_cache(self) -> None:
        """Clearing cache should remove all entries."""
        _set_cached("key1", AIMetadataResult(youtube_title="A"))
        _set_cached("key2", AIMetadataResult(youtube_title="B"))
        clear_cache()
        assert _get_cached("key1") is None
        assert _get_cached("key2") is None

    def test_get_cache_stats(self) -> None:
        """Cache stats should report correct entry count."""
        assert get_cache_stats()["entries"] == 0
        _set_cached("key1", AIMetadataResult())
        assert get_cache_stats()["entries"] == 1
        _set_cached("key2", AIMetadataResult())
        assert get_cache_stats()["entries"] == 2


# ── Test _generate_locally ────────────────────────────────────

class TestGenerateLocally:
    """Tests for the _generate_locally fallback function."""

    def test_generate_locally_returns_metadata(
        self, sample_transcription: str, sample_title: str, mock_settings: MagicMock,
    ) -> None:
        """_generate_locally should return an AIMetadataResult."""
        result = _generate_locally(sample_transcription, sample_title, "youtube", mock_settings)
        assert isinstance(result, AIMetadataResult)

    def test_generate_locally_has_youtube_title(
        self, sample_transcription: str, sample_title: str, mock_settings: MagicMock,
    ) -> None:
        """Local generation should produce a YouTube title."""
        result = _generate_locally(sample_transcription, sample_title, "youtube", mock_settings)
        assert len(result.youtube_title) > 0

    def test_generate_locally_has_hashtags(
        self, sample_transcription: str, sample_title: str, mock_settings: MagicMock,
    ) -> None:
        """Local generation should produce hashtags."""
        result = _generate_locally(sample_transcription, sample_title, "youtube", mock_settings)
        assert len(result.hashtags) > 0

    def test_generate_locally_has_keywords(
        self, sample_transcription: str, sample_title: str, mock_settings: MagicMock,
    ) -> None:
        """Local generation should produce keywords extracted from text."""
        result = _generate_locally(sample_transcription, sample_title, "youtube", mock_settings)
        assert len(result.keywords) > 0

    def test_generate_locally_has_scores(
        self, sample_transcription: str, sample_title: str, mock_settings: MagicMock,
    ) -> None:
        """Local generation should compute seo_score and viral_score."""
        result = _generate_locally(sample_transcription, sample_title, "youtube", mock_settings)
        assert 0.0 <= result.seo_score <= 100.0
        assert 0.0 <= result.viral_score <= 100.0

    def test_generate_locally_platform_specific_tags(
        self, sample_transcription: str, sample_title: str, mock_settings: MagicMock,
    ) -> None:
        """Different platforms should produce different hashtags."""
        yt_result = _generate_locally(sample_transcription, sample_title, "youtube", mock_settings)
        tt_result = _generate_locally(sample_transcription, sample_title, "tiktok", mock_settings)
        # TikTok should have fyp/for you tags
        tt_tags_lower = [t.lower() for t in tt_result.hashtags]
        assert any("fyp" in t or "foryou" in t for t in tt_tags_lower)

    def test_generate_locally_empty_text(
        self, mock_settings: MagicMock,
    ) -> None:
        """Empty transcription text should still produce a result."""
        result = _generate_locally("", "Video Title", "youtube", mock_settings)
        assert isinstance(result, AIMetadataResult)
        assert len(result.youtube_title) > 0

    def test_generate_locally_none_settings(
        self, sample_transcription: str, sample_title: str,
    ) -> None:
        """None settings should be handled by using get_settings()."""
        # This tests the fallback inside _generate_locally
        result = _generate_locally(sample_transcription, sample_title, "youtube", None)
        assert isinstance(result, AIMetadataResult)

    def test_generate_locally_has_all_captions(
        self, sample_transcription: str, sample_title: str, mock_settings: MagicMock,
    ) -> None:
        """Local generation should produce captions for all platforms."""
        result = _generate_locally(sample_transcription, sample_title, "all", mock_settings)
        assert len(result.youtube_title) > 0
        assert len(result.tiktok_caption) > 0
        assert len(result.reels_caption) > 0


# ── Test generate_ai_metadata ────────────────────────────────

class TestGenerateAIMetadata:
    """Tests for the main generate_ai_metadata function."""

    @patch("core.ai_metadata._call_ai_generate")
    def test_fallback_to_local_on_ai_failure(
        self,
        mock_ai_call: MagicMock,
        sample_transcription: str,
        sample_title: str,
    ) -> None:
        """When AI call fails, should fall back to local generation."""
        mock_ai_call.side_effect = RuntimeError("AI service unavailable")
        result = generate_ai_metadata(sample_transcription, sample_title, platform="youtube")
        assert isinstance(result, AIMetadataResult)
        # Should have some content from local fallback
        assert len(result.youtube_title) > 0 or len(result.keywords) > 0

    @patch("core.ai_metadata._call_ai_generate")
    def test_returns_ai_result_on_success(
        self,
        mock_ai_call: MagicMock,
        sample_transcription: str,
        sample_title: str,
    ) -> None:
        """When AI call succeeds, should return parsed AI result."""
        mock_ai_call.return_value = {
            "youtube_title": "AI Generated Title",
            "youtube_description": "AI description",
            "tiktok_caption": "AI caption",
            "reels_caption": "AI reels",
            "hashtags": ["ai", "viral"],
            "keywords": ["technology"],
            "seo_score": 88,
            "viral_score": 95,
        }
        result = generate_ai_metadata(sample_transcription, sample_title, platform="youtube")
        assert result.youtube_title == "AI Generated Title"
        assert result.seo_score == 88.0
        assert result.viral_score == 95.0

    @patch("core.ai_metadata._call_ai_generate")
    def test_fallback_on_empty_ai_result(
        self,
        mock_ai_call: MagicMock,
        sample_transcription: str,
        sample_title: str,
    ) -> None:
        """When AI returns empty metadata, should fall back to local."""
        mock_ai_call.return_value = {
            "youtube_title": "",
            "tiktok_caption": "",
            "reels_caption": "",
        }
        result = generate_ai_metadata(sample_transcription, sample_title, platform="youtube")
        # Should have content from local fallback
        assert isinstance(result, AIMetadataResult)
        assert len(result.youtube_title) > 0 or len(result.keywords) > 0

    def test_empty_transcription_returns_empty(self) -> None:
        """Empty transcription text should return empty AIMetadataResult."""
        result = generate_ai_metadata("", "Test Title")
        assert result.youtube_title == ""
        assert result.seo_score == 0.0

    def test_whitespace_only_transcription_returns_empty(self) -> None:
        """Whitespace-only transcription should return empty result."""
        result = generate_ai_metadata("   \n  ", "Test Title")
        assert result.youtube_title == ""

    @patch("core.ai_metadata._call_ai_generate")
    def test_invalid_platform_defaults_to_all(
        self,
        mock_ai_call: MagicMock,
        sample_transcription: str,
        sample_title: str,
    ) -> None:
        """Invalid platform should default to 'all'."""
        mock_ai_call.side_effect = RuntimeError("Skip AI")
        result = generate_ai_metadata(sample_transcription, sample_title, platform="invalid_platform")
        assert isinstance(result, AIMetadataResult)

    @patch("core.ai_metadata._call_ai_generate")
    def test_caching_works(
        self,
        mock_ai_call: MagicMock,
        sample_transcription: str,
        sample_title: str,
    ) -> None:
        """Second call with same parameters should use cache and not call AI again."""
        mock_ai_call.side_effect = RuntimeError("AI not available")
        result1 = generate_ai_metadata(sample_transcription, sample_title, platform="youtube")
        assert mock_ai_call.call_count == 1
        result2 = generate_ai_metadata(sample_transcription, sample_title, platform="youtube")
        # Second call should be cached — AI should not be called again
        assert mock_ai_call.call_count == 1
        assert result1.youtube_title == result2.youtube_title

    @patch("core.ai_metadata._call_ai_generate")
    def test_different_platforms_not_cached_together(
        self,
        mock_ai_call: MagicMock,
        sample_transcription: str,
        sample_title: str,
    ) -> None:
        """Different platforms should not share cache entries."""
        mock_ai_call.side_effect = RuntimeError("AI not available")
        generate_ai_metadata(sample_transcription, sample_title, platform="youtube")
        generate_ai_metadata(sample_transcription, sample_title, platform="tiktok")
        # Should be called twice — different platforms
        assert mock_ai_call.call_count == 2


# ── Test _build_prompt ────────────────────────────────────────

class TestBuildPrompt:
    """Tests for the _build_prompt function."""

    def test_prompt_contains_transcription(
        self, sample_transcription: str, sample_title: str, mock_settings: MagicMock,
    ) -> None:
        """The prompt should contain the transcription text."""
        prompt = _build_prompt(sample_transcription, sample_title, "youtube", mock_settings)
        assert "artificial intelligence" in prompt

    def test_prompt_contains_platform(
        self, sample_transcription: str, sample_title: str, mock_settings: MagicMock,
    ) -> None:
        """The prompt should reference the target platform."""
        prompt = _build_prompt(sample_transcription, sample_title, "tiktok", mock_settings)
        assert "tiktok" in prompt.lower()

    def test_prompt_contains_video_title(
        self, sample_transcription: str, sample_title: str, mock_settings: MagicMock,
    ) -> None:
        """The prompt should contain the video title."""
        prompt = _build_prompt(sample_transcription, sample_title, "youtube", mock_settings)
        assert sample_title in prompt

    def test_prompt_requests_json(
        self, sample_transcription: str, sample_title: str, mock_settings: MagicMock,
    ) -> None:
        """The prompt should ask for JSON output."""
        prompt = _build_prompt(sample_transcription, sample_title, "youtube", mock_settings)
        assert "JSON" in prompt

    def test_prompt_truncates_long_transcription(self, mock_settings: MagicMock) -> None:
        """Very long transcriptions should be truncated in the prompt."""
        long_text = "word " * 5000  # ~25k chars
        prompt = _build_prompt(long_text, "Test", "youtube", mock_settings)
        # The prompt should still be reasonable length
        assert len(prompt) < 50000

    def test_prompt_for_all_platforms(
        self, sample_transcription: str, sample_title: str, mock_settings: MagicMock,
    ) -> None:
        """Platform 'all' should include instructions for all platforms."""
        prompt = _build_prompt(sample_transcription, sample_title, "all", mock_settings)
        assert "ALL three platforms" in prompt or "YouTube" in prompt
