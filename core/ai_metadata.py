"""
core/ai_metadata.py — AI-powered metadata generation for YouTube Shorts, TikTok,
and Instagram Reels using z-ai-web-dev-sdk.

Generates viral titles, descriptions, hashtags, and captions via a Node.js helper
script that calls the z-ai-web-dev-sdk. Falls back to local keyword extraction
(from metadata_generator.py) if the AI call fails.

Features:
- Platform-specific formatting (YouTube, TikTok, Reels)
- Smart prompt engineering for viral content
- Caching to avoid re-generating for the same content
- Graceful fallback to local keyword extraction
- SEO and viral scoring
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.settings import Settings, get_settings
from utils.logger import get_logger

logger = get_logger("ai_metadata")

# Path to the Node.js helper script
_AI_GENERATE_SCRIPT: Path = Path(__file__).resolve().parent.parent / "scripts" / "ai_generate.js"

# Maximum prompt length (characters) — keeps payloads manageable
_MAX_PROMPT_LENGTH: int = 16000

# Timeout for the Node.js subprocess (seconds)
_SUBPROCESS_TIMEOUT: int = 120

# ── Stopwords (reused from metadata_generator patterns) ──────

_STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "if",
    "in", "into", "is", "it", "its", "itself", "just", "let", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves", "also", "like", "even", "well", "back",
    "still", "way", "take", "come", "make", "know", "think", "going",
    "really", "thing", "things", "something", "much", "many", "go", "see",
    "look", "say", "said", "one", "two", "first", "new", "want", "need",
    "right", "now", "actually", "basically", "literally", "probably",
    "maybe", "kind", "sort", "quite", "pretty", "already", "always",
    "never", "ever", "around", "since", "every", "another", "next", "last",
    "long", "great", "little", "big", "old", "people", "good", "time",
    "been", "being", "got", "getting", "yeah", "okay", "oh",
    "um", "uh", "hmm",
}


# ── Data Classes ──────────────────────────────────────────────


@dataclass
class AIMetadataResult:
    """AI-generated metadata for all platforms."""

    youtube_title: str = ""
    youtube_description: str = ""
    tiktok_caption: str = ""
    reels_caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    seo_score: float = 0.0
    viral_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for JSON serialisation."""
        return {
            "youtube_title": self.youtube_title,
            "youtube_description": self.youtube_description,
            "tiktok_caption": self.tiktok_caption,
            "reels_caption": self.reels_caption,
            "hashtags": self.hashtags,
            "keywords": self.keywords,
            "seo_score": self.seo_score,
            "viral_score": self.viral_score,
        }


# ── In-Memory Cache ───────────────────────────────────────────

_cache_lock = threading.Lock()
_metadata_cache: dict[str, AIMetadataResult] = {}


def _compute_cache_key(transcription_text: str, video_title: str, platform: str, settings_hash: str) -> str:
    """Compute a deterministic cache key from input parameters.

    Args:
        transcription_text: The transcription text.
        video_title: The video title.
        platform: Target platform identifier.
        settings_hash: Hash of relevant settings.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    raw = f"{transcription_text}||{video_title}||{platform}||{settings_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_cached(key: str) -> Optional[AIMetadataResult]:
    """Retrieve a cached result if available.

    Args:
        key: Cache key string.

    Returns:
        Cached AIMetadataResult, or None if not cached.
    """
    with _cache_lock:
        return _metadata_cache.get(key)


def _set_cached(key: str, result: AIMetadataResult) -> None:
    """Store a result in the cache.

    Args:
        key: Cache key string.
        result: AIMetadataResult to cache.
    """
    with _cache_lock:
        _metadata_cache[key] = result


def clear_cache() -> None:
    """Clear the metadata cache."""
    with _cache_lock:
        _metadata_cache.clear()
    logger.info("AI metadata cache cleared")


def get_cache_stats() -> dict[str, int]:
    """Get cache statistics.

    Returns:
        Dictionary with 'entries' count.
    """
    with _cache_lock:
        return {"entries": len(_metadata_cache)}


# ── Prompt Builder ────────────────────────────────────────────


def _build_prompt(
    transcription_text: str,
    video_title: str,
    platform: str,
    settings: Settings,
) -> str:
    """Build a smart prompt for the AI to generate viral metadata.

    Includes platform-specific formatting rules and asks for structured
    JSON output with titles, descriptions, captions, hashtags, and scores.

    Args:
        transcription_text: The transcribed speech text.
        video_title: Original video title.
        platform: Target platform ('youtube', 'tiktok', 'reels', or 'all').
        settings: Application settings.

    Returns:
        Prompt string for the AI.
    """
    # Truncate transcription to fit within prompt limits
    max_text = 8000
    truncated_text = transcription_text[:max_text]
    if len(transcription_text) > max_text:
        truncated_text += "... [truncated]"

    hashtag_count = settings.METADATA_HASHTAG_COUNT

    platform_instructions = {
        "youtube": (
            "YouTube Shorts: Create a catchy title (30-65 chars), an engaging description "
            "(100-300 chars with key points and hashtags), and relevant hashtags."
        ),
        "tiktok": (
            "TikTok: Create a snappy caption (under 150 chars, with emojis and hashtags), "
            "optimized for the For You page. Use trending TikTok hashtags."
        ),
        "reels": (
            "Instagram Reels: Create an engaging caption (under 2200 chars, with line breaks "
            "and emojis), use a mix of popular and niche hashtags."
        ),
        "all": (
            "Generate metadata for ALL three platforms:\n"
            "1. YouTube Shorts: catchy title (30-65 chars) + description (100-300 chars)\n"
            "2. TikTok: snappy caption (under 150 chars, with emojis)\n"
            "3. Instagram Reels: engaging caption (under 2200 chars, with emojis)\n"
            "Each platform should have its own optimized hashtags."
        ),
    }

    platform_instruction = platform_instructions.get(platform, platform_instructions["all"])

    prompt = f"""You are a viral social media metadata expert. Generate optimized metadata for short-form video content.

ORIGINAL VIDEO TITLE: {video_title}

TRANSCRIPTION:
{truncated_text}

PLATFORM: {platform}
INSTRUCTIONS: {platform_instruction}

REQUIREMENTS:
- Generate {hashtag_count} relevant hashtags (without # prefix)
- Include 5-10 SEO keywords extracted from the content
- Rate SEO effectiveness (0-100)
- Rate viral potential (0-100) based on hook strength, emotional appeal, and trend alignment
- Titles/captions should be attention-grabbing and click-worthy
- Use appropriate emojis for TikTok and Reels captions
- Hashtags should mix broad/trending with niche/specific tags

Respond with ONLY valid JSON in this exact format:
{{
  "youtube_title": "...",
  "youtube_description": "...",
  "tiktok_caption": "...",
  "reels_caption": "...",
  "hashtags": ["tag1", "tag2", ...],
  "keywords": ["keyword1", "keyword2", ...],
  "seo_score": 85,
  "viral_score": 92
}}"""

    return prompt


# ── AI Call via Subprocess ────────────────────────────────────


def _call_ai_generate(prompt: str) -> dict[str, Any]:
    """Call the Node.js helper script to generate metadata via z-ai-web-dev-sdk.

    Runs the ai_generate.js script as a subprocess, passing the prompt
    as a command-line argument, and parses the JSON output.

    Args:
        prompt: The prompt string to send to the AI.

    Returns:
        Parsed JSON dictionary from the AI response.

    Raises:
        RuntimeError: If the script fails or returns invalid output.
    """
    if not _AI_GENERATE_SCRIPT.exists():
        raise RuntimeError(
            f"AI helper script not found: {_AI_GENERATE_SCRIPT}"
        )

    # Find node executable
    node_path = os.environ.get("NODE_PATH", "node")

    cmd = [node_path, str(_AI_GENERATE_SCRIPT), "--prompt", prompt]

    logger.debug("Calling AI generate script: %s", _AI_GENERATE_SCRIPT.name)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
            cwd=str(_AI_GENERATE_SCRIPT.parent.parent),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"AI generate script timed out after {_SUBPROCESS_TIMEOUT}s"
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Node.js executable not found: {node_path}. "
            "Install Node.js or set NODE_PATH environment variable."
        )

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(
            f"AI generate script failed (exit code {result.returncode}): {stderr}"
        )

    # Parse stdout — may contain multiple lines, find the JSON
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("AI generate script returned empty output")

    # Try each line as JSON (the script outputs one JSON line)
    for line in stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            return parsed
        except json.JSONDecodeError:
            continue

    # If no line parsed, try the whole output
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Could not parse AI generate output as JSON: {stdout[:500]}"
        )


# ── Response Parser ───────────────────────────────────────────


def _parse_ai_response(response: dict[str, Any]) -> AIMetadataResult:
    """Parse the AI response into an AIMetadataResult dataclass.

    Handles both direct data responses and wrapped responses (with 'data' key).
    Falls back gracefully for missing fields.

    Args:
        response: Parsed JSON dictionary from the AI or the Node.js script.

    Returns:
        AIMetadataResult with extracted fields.
    """
    # The Node.js script wraps successful AI output in { data: ..., parsed: true }
    data = response
    if "data" in response and isinstance(response["data"], dict):
        data = response["data"]
    elif "raw_text" in response:
        # AI returned non-JSON text — try to extract JSON from it
        raw = response["raw_text"]
        try:
            # Try to find JSON object in the raw text
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Could not extract JSON from AI raw response")
            return AIMetadataResult()

    # If there's an error in the response, log it and return empty
    if "error" in data and "data" not in response:
        error_msg = data.get("error", "Unknown error")
        logger.error("AI generate returned error: %s", error_msg)
        return AIMetadataResult()

    return AIMetadataResult(
        youtube_title=str(data.get("youtube_title", ""))[:200],
        youtube_description=str(data.get("youtube_description", ""))[:2000],
        tiktok_caption=str(data.get("tiktok_caption", ""))[:500],
        reels_caption=str(data.get("reels_caption", ""))[:2200],
        hashtags=_clean_hashtags(data.get("hashtags", [])),
        keywords=_clean_keywords(data.get("keywords", [])),
        seo_score=_clamp_score(data.get("seo_score", 0)),
        viral_score=_clamp_score(data.get("viral_score", 0)),
    )


def _clamp_score(value: Any) -> float:
    """Clamp a score value to the 0-100 range.

    Args:
        value: Numeric score value (may be int, float, or string).

    Returns:
        Float clamped to [0.0, 100.0].
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, round(score, 1)))


def _clean_hashtags(hashtags: Any) -> list[str]:
    """Clean and validate a list of hashtags.

    Strips # prefixes, removes empty/duplicate entries, and limits count.

    Args:
        hashtags: Raw hashtag list (may contain strings with # prefix).

    Returns:
        Cleaned list of hashtag strings without # prefix.
    """
    if not isinstance(hashtags, list):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()

    for tag in hashtags[:30]:  # Cap at 30
        tag_str = str(tag).strip().lstrip("#").strip()
        if len(tag_str) >= 2 and tag_str.lower() not in seen:
            seen.add(tag_str.lower())
            cleaned.append(tag_str)

    return cleaned[:20]  # Return at most 20


def _clean_keywords(keywords: Any) -> list[str]:
    """Clean and validate a list of keywords.

    Removes empty entries, deduplicates, and limits count.

    Args:
        keywords: Raw keyword list.

    Returns:
        Cleaned list of keyword strings.
    """
    if not isinstance(keywords, list):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()

    for kw in keywords[:30]:
        kw_str = str(kw).strip().lower()
        if len(kw_str) >= 2 and kw_str not in seen:
            seen.add(kw_str)
            cleaned.append(kw_str)

    return cleaned[:15]


# ── Local Fallback Generation ─────────────────────────────────


def _generate_locally(
    transcription_text: str,
    video_title: str,
    platform: str,
    settings: Settings,
) -> AIMetadataResult:
    """Generate metadata locally using keyword extraction as a fallback.

    Reuses keyword extraction logic from metadata_generator.py patterns.
    Used when the AI service is unavailable or returns an error.

    Args:
        transcription_text: The transcribed speech text.
        video_title: Original video title.
        platform: Target platform identifier.
        settings: Application settings.

    Returns:
        AIMetadataResult with locally-generated metadata.
    """
    if settings is None:
        settings = get_settings()

    logger.info("Falling back to local metadata generation")

    # ── Extract keywords from transcription ────────────
    words: list[str] = []
    for word in re.split(r"\s+", transcription_text.lower()):
        clean_word = re.sub(r"[^\w]", "", word)
        if len(clean_word) >= 2 and clean_word not in _STOPWORDS:
            words.append(clean_word)

    # Also extract from the video title
    title_words: list[str] = []
    for word in re.split(r"\s+", video_title.lower()):
        clean_word = re.sub(r"[^\w]", "", word)
        if len(clean_word) >= 2 and clean_word not in _STOPWORDS:
            title_words.append(clean_word)

    # Combine with title boost
    word_freq: Counter = Counter(words)
    for tw in title_words:
        word_freq[tw] = word_freq.get(tw, 0) + 5  # Title boost

    # Top keywords
    top_keywords = [kw for kw, _ in word_freq.most_common(settings.METADATA_MAX_KEYWORDS)]

    # ── Generate platform-specific content ─────────────
    primary_keyword = top_keywords[0].title() if top_keywords else "Shorts"

    # YouTube title — aim for 30-65 chars
    yt_title = f"{primary_keyword} — You Won't Believe This!"
    if len(yt_title) > 65:
        yt_title = yt_title[:62] + "..."

    # YouTube description
    key_points = "\n".join(f"  • {kw.title()}" for kw in top_keywords[:5])
    yt_description = (
        f"Discover the power of {primary_keyword}!\n\n"
        f"Key Topics:\n{key_points}\n\n"
        f"Subscribe for more shorts! \U0001f514"
    )

    # TikTok caption — short and punchy
    tiktok_caption = f"Wait for it... \U0001f525 #{primary_keyword}"

    # Reels caption — engaging with line breaks
    reels_caption = (
        f"This {primary_keyword} moment though \u2764\ufe0f\n"
        f"Save this for later! \U0001f516\n"
        f"Follow for more {primary_keyword.lower()} content!"
    )

    # Hashtags — combine keyword, category, and platform tags
    hashtag_count = settings.METADATA_HASHTAG_COUNT
    hashtags: list[str] = []

    # Keyword-based hashtags
    for kw in top_keywords[:5]:
        tag = kw.replace(" ", "").replace("_", "")
        if len(tag) >= 3:
            hashtags.append(tag)

    # Platform-specific hashtags
    platform_tags: dict[str, list[str]] = {
        "youtube": ["Shorts", "YouTubeShorts", "Youtuber"],
        "tiktok": ["fyp", "foryou", "viral", "trending"],
        "reels": ["Reels", "ReelsInstagram", "InstaReels"],
        "all": ["Shorts", "fyp", "viral", "trending"],
    }
    hashtags.extend(platform_tags.get(platform, platform_tags["all"]))

    # Deduplicate
    seen: set[str] = set()
    unique_hashtags: list[str] = []
    for tag in hashtags:
        if tag.lower() not in seen:
            seen.add(tag.lower())
            unique_hashtags.append(tag)

    # ── Score locally ──────────────────────────────────
    seo_score = _compute_local_seo_score(yt_title, yt_description, unique_hashtags, top_keywords)
    viral_score = _compute_local_viral_score(transcription_text, yt_title)

    return AIMetadataResult(
        youtube_title=yt_title,
        youtube_description=yt_description,
        tiktok_caption=tiktok_caption,
        reels_caption=reels_caption,
        hashtags=unique_hashtags[:hashtag_count],
        keywords=top_keywords,
        seo_score=seo_score,
        viral_score=viral_score,
    )


def _compute_local_seo_score(
    title: str,
    description: str,
    hashtags: list[str],
    keywords: list[str],
) -> float:
    """Compute a rough SEO score based on local heuristics.

    Args:
        title: Generated title.
        description: Generated description.
        hashtags: Generated hashtags.
        keywords: Target keywords.

    Returns:
        SEO score between 0 and 100.
    """
    score = 0.0

    # Title length (30-65 chars is ideal)
    title_len = len(title)
    if 30 <= title_len <= 65:
        score += 30.0
    elif 20 <= title_len <= 80:
        score += 20.0
    else:
        score += 5.0

    # Keywords in title
    title_lower = title.lower()
    keywords_in_title = sum(1 for kw in keywords[:5] if kw in title_lower)
    score += keywords_in_title * 10.0

    # Description length
    if len(description) >= 100:
        score += 20.0
    elif len(description) >= 50:
        score += 10.0

    # Hashtags
    if 3 <= len(hashtags) <= 15:
        score += 10.0
    elif len(hashtags) >= 2:
        score += 5.0

    return min(100.0, round(score, 1))


def _compute_local_viral_score(
    transcription_text: str,
    title: str,
) -> float:
    """Compute a rough viral potential score based on local heuristics.

    Looks for emotional hooks, questions, power words, and engagement
    triggers in the text and title.

    Args:
        transcription_text: The transcription text.
        title: Generated title.

    Returns:
        Viral score between 0 and 100.
    """
    score = 30.0  # Base score

    combined = f"{transcription_text} {title}".lower()

    # Hook words
    hook_words = {
        "secret", "hidden", "nobody", "never", "shocking", "unbelievable",
        "amazing", "incredible", "insane", "crazy", "mind", "blown",
        "you won't", "wait", "watch", "before", "after", "must",
    }
    hook_matches = sum(1 for hw in hook_words if hw in combined)
    score += min(30.0, hook_matches * 6.0)

    # Questions increase engagement
    question_count = combined.count("?")
    score += min(15.0, question_count * 5.0)

    # Exclamations show energy
    exclamation_count = combined.count("!")
    score += min(10.0, exclamation_count * 3.0)

    # Emoji presence (rough proxy)
    emoji_count = len(re.findall(r"[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff\U00002702-\U000027b0]", combined))
    score += min(15.0, emoji_count * 3.0)

    return min(100.0, round(score, 1))


# ── Public API ────────────────────────────────────────────────


def generate_ai_metadata(
    transcription_text: str,
    video_title: str = "",
    platform: str = "all",
    settings: Settings | None = None,
) -> AIMetadataResult:
    """Generate AI-powered metadata for YouTube Shorts, TikTok, and Instagram Reels.

    Builds a smart prompt that asks the AI to generate viral titles, descriptions,
    hashtags, and captions with platform-specific formatting. Falls back to local
    keyword extraction if the AI call fails. Results are cached to avoid
    re-generating for the same content.

    Args:
        transcription_text: The transcribed speech text from the video.
        video_title: Original video title (used for context).
        platform: Target platform — 'youtube', 'tiktok', 'reels', or 'all'.
        settings: Optional Settings override.

    Returns:
        AIMetadataResult with AI-generated (or fallback) metadata including
        youtube_title, youtube_description, tiktok_caption, reels_caption,
        hashtags, keywords, seo_score, and viral_score.
    """
    if settings is None:
        settings = get_settings()

    # Validate platform
    valid_platforms = {"youtube", "tiktok", "reels", "all"}
    if platform not in valid_platforms:
        logger.warning("Invalid platform '%s', defaulting to 'all'", platform)
        platform = "all"

    # Handle empty transcription
    if not transcription_text or not transcription_text.strip():
        logger.warning("Empty transcription text, returning empty metadata")
        return AIMetadataResult()

    # ── Check cache ────────────────────────────────────
    settings_hash = hashlib.md5(
        f"{settings.METADATA_HASHTAG_COUNT}|{settings.METADATA_MAX_KEYWORDS}|{settings.METADATA_LANGUAGE}".encode()
    ).hexdigest()

    cache_key = _compute_cache_key(transcription_text, video_title, platform, settings_hash)
    cached = _get_cached(cache_key)
    if cached is not None:
        logger.info("Returning cached AI metadata (key=%s...)", cache_key[:12])
        return cached

    # ── Build prompt ───────────────────────────────────
    prompt = _build_prompt(transcription_text, video_title, platform, settings)

    if len(prompt) > _MAX_PROMPT_LENGTH:
        logger.warning(
            "Prompt too long (%d chars), truncating transcription",
            len(prompt),
        )
        # Rebuild with shorter transcription
        shorter_text = transcription_text[:4000]
        prompt = _build_prompt(shorter_text, video_title, platform, settings)

    # ── Call AI ────────────────────────────────────────
    result: AIMetadataResult
    try:
        logger.info(
            "Generating AI metadata for '%s' (platform=%s, text_len=%d)",
            video_title[:50] if video_title else "untitled",
            platform,
            len(transcription_text),
        )

        raw_response = _call_ai_generate(prompt)
        result = _parse_ai_response(raw_response)

        # Validate we got meaningful results
        if not result.youtube_title and not result.tiktok_caption and not result.reels_caption:
            logger.warning("AI returned empty metadata, falling back to local generation")
            result = _generate_locally(transcription_text, video_title, platform, settings)

    except RuntimeError as exc:
        logger.error("AI generation failed: %s — falling back to local", exc)
        result = _generate_locally(transcription_text, video_title, platform, settings)
    except Exception as exc:
        logger.error("Unexpected error during AI generation: %s — falling back to local", exc)
        result = _generate_locally(transcription_text, video_title, platform, settings)

    # ── Cache the result ───────────────────────────────
    _set_cached(cache_key, result)

    logger.info(
        "AI metadata generated: yt_title='%s' | seo=%.0f | viral=%.0f | hashtags=%d | keywords=%d",
        result.youtube_title[:40] if result.youtube_title else "(none)",
        result.seo_score,
        result.viral_score,
        len(result.hashtags),
        len(result.keywords),
    )

    return result
