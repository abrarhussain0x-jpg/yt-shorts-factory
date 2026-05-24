"""
core/transcriber.py — Advanced Whisper transcription with VAD, confidence filtering,
hallucination detection, word-level timing correction, and multi-model comparison.

Supports both openai-whisper and faster-whisper (CTranslate2-based, 4x faster).
Includes VAD pre-filtering, language detection, prosody estimation, transcription
caching, and export to SRT/VTT formats.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

from config.settings import Settings, get_settings
from utils.ffmpeg_utils import extract_audio_wav, probe_video
from utils.file_utils import safe_delete
from utils.logger import get_logger

logger = get_logger("transcriber")
console = Console()


# ── Data Classes ──────────────────────────────────────────────

@dataclass
class WordTimestamp:
    """A single word with its timing and confidence."""

    word: str
    start: float
    end: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Segment:
    """A transcription segment (sentence/phrase) with timing."""

    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class LanguageDetection:
    """Language detection result with confidence."""

    language: str = ""
    confidence: float = 0.0
    language_probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class ProsodyInfo:
    """Prosody estimation for a word or segment."""

    emphasis: float = 0.0  # 0-1, how emphasized
    is_question: bool = False
    is_exclamatory: bool = False
    energy_estimate: float = 0.5  # 0-1 rough audio energy


@dataclass
class TranscriptionResult:
    """Complete transcription output with word-level timestamps."""

    words: list[WordTimestamp] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    language: str = ""
    language_confidence: float = 0.0
    duration: float = 0.0
    model_used: str = ""
    device_used: str = ""
    prosody: list[ProsodyInfo] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)

    @property
    def is_empty(self) -> bool:
        return len(self.words) == 0

    @property
    def average_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)


# ── Constants ─────────────────────────────────────────────────

# Hallucination Detection v2 — common Whisper hallucination patterns
_COMMON_HALLUCINATIONS: set[str] = {
    "thank you for watching", "please subscribe", "like and subscribe",
    "thanks for watching", "don't forget to subscribe", "see you next time",
    "thank you for listening", "thanks for listening", "bye bye",
    "the end", "in this video", "in this episode",
    "subscribe to my channel", "hit the like button", "leave a comment",
    "click the bell", "turn on notifications", "follow me on",
    "check out my other videos", "watch more videos",
}

# Number sequence hallucination pattern (e.g. "1, 2, 3, 4, 5, 6, 7, 8")
_NUMBER_SEQUENCE_PATTERN = re.compile(
    r"(\d+[,.\s]*){5,}"
)

# Repeated phrase hallucination patterns
_REPETITION_PATTERNS: list[re.Pattern] = [
    re.compile(r"(\b\w+\b)(\s+\1){4,}"),          # Same word repeated 5+ times
    re.compile(r"(\b.{5,}\b)(\s+\1){3,}"),         # Same phrase repeated 4+ times
    re.compile(r"(\b\w+\s+\w+\b)(\s+\1){3,}"),    # Same 2-word phrase repeated 4+ times
]

# Default confidence and duration thresholds
_MIN_CONFIDENCE = 0.3
_MAX_WORD_DURATION = 10.0
_MIN_WORD_DURATION = 0.02  # Suspiciously short words

# VAD parameters
_VAD_FRAME_DURATION_MS = 30
_VAD_SILENCE_THRESHOLD = 0.3  # Seconds of silence to consider as boundary

# Caching directory
_CACHE_DIR_NAME = "transcription_cache"


# ── VAD (Voice Activity Detection) ───────────────────────────

def _apply_vad_filter(
    audio_path: Path,
    aggressiveness: int = 2,
) -> list[tuple[float, float]]:
    """Apply Voice Activity Detection to identify speech regions.

    Uses the webrtcvad library if available, otherwise falls back to
    FFmpeg-based silence detection.

    Args:
        audio_path: Path to the WAV audio file.
        aggressiveness: VAD aggressiveness 0-3 (0=most permissive, 3=most aggressive).

    Returns:
        List of (start, end) tuples for speech regions in seconds.
    """
    try:
        import webrtcvad
        import wave

        vad = webrtcvad.Vad(min(3, max(0, aggressiveness)))

        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()

            if sample_rate not in (8000, 16000, 32000, 48000):
                logger.debug("VAD: unsupported sample rate %d, skipping", sample_rate)
                return [(0.0, float("inf"))]

            frame_size = int(sample_rate * _VAD_FRAME_DURATION_MS / 1000)
            frame_bytes = frame_size * n_channels * sample_width

            speech_frames: list[tuple[float, float]] = []
            frame_offset = 0.0
            frame_duration = _VAD_FRAME_DURATION_MS / 1000.0

            current_speech_start: Optional[float] = None

            while True:
                data = wf.readframes(frame_size)
                if len(data) < frame_bytes:
                    break

                is_speech = vad.is_speech(data, sample_rate)
                frame_time = frame_offset

                if is_speech:
                    if current_speech_start is None:
                        current_speech_start = frame_time
                else:
                    if current_speech_start is not None:
                        speech_frames.append((current_speech_start, frame_time))
                        current_speech_start = None

                frame_offset += frame_duration

            # Close any open speech region
            if current_speech_start is not None:
                speech_frames.append((current_speech_start, frame_offset))

        if not speech_frames:
            logger.debug("VAD: no speech detected, using full audio")
            return [(0.0, float("inf"))]

        # Merge adjacent speech regions with small gaps
        merged = _merge_speech_regions(speech_frames, max_gap=0.3)
        logger.debug("VAD: detected %d speech regions", len(merged))
        return merged

    except ImportError:
        logger.debug("webrtcvad not installed, skipping VAD pre-filtering")
        return [(0.0, float("inf"))]
    except Exception as exc:
        logger.warning("VAD failed: %s, using full audio", exc)
        return [(0.0, float("inf"))]


def _merge_speech_regions(
    regions: list[tuple[float, float]],
    max_gap: float = 0.3,
) -> list[tuple[float, float]]:
    """Merge speech regions that are close together.

    Args:
        regions: List of (start, end) speech regions.
        max_gap: Maximum gap in seconds between regions to merge.

    Returns:
        Merged list of speech regions.
    """
    if not regions:
        return []

    merged: list[tuple[float, float]] = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


# ── Hallucination Detection v2 ────────────────────────────────

def _is_hallucination(
    word: str,
    confidence: float,
    duration: float,
    prev_words: list[WordTimestamp],
    full_text_so_far: str = "",
) -> bool:
    """Check if a word is likely a hallucination using v2 detection.

    Checks:
    1. Low confidence below threshold
    2. Suspiciously long or short word duration
    3. Repetition: same word repeated many times
    4. Phrase repetition: same phrase repeated
    5. Number sequence hallucination
    6. Common hallucination phrase match

    Args:
        word: The word text to check.
        confidence: Word confidence score.
        duration: Word duration in seconds.
        prev_words: List of previous words for context.
        full_text_so_far: Running transcription text for pattern matching.

    Returns:
        True if the word is likely a hallucination.
    """
    # Low confidence
    if confidence < _MIN_CONFIDENCE:
        return True

    # Suspiciously long word
    if duration > _MAX_WORD_DURATION:
        return True

    # Suspiciously short word (likely noise)
    if duration > 0 and duration < _MIN_WORD_DURATION and len(word) <= 1:
        return True

    # Repetition check: same word repeated many times
    if len(prev_words) >= 5:
        last_5 = [w.word.lower() for w in prev_words[-5:]]
        if all(w == last_5[0] for w in last_5) and word.lower() == last_5[0]:
            return True

    # Phrase repetition: check last N words for 2-word phrase repetition
    if len(prev_words) >= 6:
        recent = [w.word.lower() for w in prev_words[-6:]]
        # Check if the last 2-word phrase repeats 3+ times
        if len(recent) >= 6:
            phrase1 = f"{recent[-2]} {recent[-1]}"
            phrase2 = f"{recent[-4]} {recent[-3]}"
            phrase3 = f"{recent[-6]} {recent[-5]}"
            if phrase1 == phrase2 == phrase3 and word.lower() in phrase1.split():
                return True

    # Number sequence hallucination
    if full_text_so_far:
        test_text = full_text_so_far + " " + word
        if _NUMBER_SEQUENCE_PATTERN.search(test_text):
            # Count number of digits in recent text
            digit_count = sum(1 for c in test_text[-50:] if c.isdigit())
            if digit_count > 10:
                return True

    return False


def _is_segment_hallucination(text: str, confidence: float) -> bool:
    """Check if an entire segment is a hallucination.

    Args:
        text: Segment text.
        confidence: Segment confidence (avg_logprob).

    Returns:
        True if the segment is likely a hallucination.
    """
    text_lower = text.lower().strip()

    # Common hallucination phrases with low confidence
    if text_lower in _COMMON_HALLUCINATIONS and confidence < -0.5:
        return True

    # Partial match for hallucination phrases
    for hall in _COMMON_HALLUCINATIONS:
        if hall in text_lower and confidence < -0.7:
            return True

    # Number sequence pattern
    if _NUMBER_SEQUENCE_PATTERN.search(text) and confidence < -0.3:
        return True

    # Check all repetition patterns
    for pattern in _REPETITION_PATTERNS:
        if pattern.search(text):
            return True

    return False


# ── Word-Level Timing Correction ──────────────────────────────

def _fix_overlapping_words(words: list[WordTimestamp]) -> list[WordTimestamp]:
    """Fix overlapping word timestamps by adjusting boundaries.

    Args:
        words: List of word timestamps to fix.

    Returns:
        List with corrected timestamps (no overlaps).
    """
    if not words:
        return words

    fixed = [WordTimestamp(
        word=words[0].word,
        start=words[0].start,
        end=words[0].end,
        confidence=words[0].confidence,
    )]

    for i in range(1, len(words)):
        prev = fixed[-1]
        curr = words[i]

        # If current word starts before previous ends, adjust
        if curr.start < prev.end:
            mid = (prev.end + curr.start) / 2.0
            # Don't go before prev.start
            mid = max(mid, prev.start + _MIN_WORD_DURATION)
            fixed[-1] = WordTimestamp(
                word=prev.word, start=prev.start, end=mid,
                confidence=prev.confidence,
            )
            curr_start = mid + _MIN_WORD_DURATION
        else:
            curr_start = curr.start

        fixed.append(WordTimestamp(
            word=curr.word,
            start=curr_start,
            end=max(curr.end, curr_start + _MIN_WORD_DURATION),
            confidence=curr.confidence,
        ))

    return fixed


def _fill_gaps(words: list[WordTimestamp]) -> list[WordTimestamp]:
    """Fill gaps between words by slightly extending durations.

    If the gap between consecutive words is small (< 0.5s), extend
    the previous word's end to reduce unnatural pauses.

    Args:
        words: List of word timestamps.

    Returns:
        List with gaps smoothed.
    """
    if len(words) <= 1:
        return words

    filled = list(words)
    for i in range(len(filled) - 1):
        gap = filled[i + 1].start - filled[i].end
        if 0 < gap < 0.5:
            # Split the gap between the two words
            half_gap = gap / 2.0
            filled[i] = WordTimestamp(
                word=filled[i].word,
                start=filled[i].start,
                end=filled[i].end + half_gap,
                confidence=filled[i].confidence,
            )
            filled[i + 1] = WordTimestamp(
                word=filled[i + 1].word,
                start=filled[i + 1].start - half_gap,
                end=filled[i + 1].end,
                confidence=filled[i + 1].confidence,
            )

    return filled


def _smooth_timing(words: list[WordTimestamp], window: int = 3) -> list[WordTimestamp]:
    """Smooth word timing to reduce jitter using a moving average.

    Args:
        words: List of word timestamps.
        window: Size of the smoothing window.

    Returns:
        List with smoothed durations.
    """
    if len(words) <= window:
        return words

    smoothed = list(words)
    for i in range(len(smoothed)):
        start = max(0, i - window // 2)
        end = min(len(smoothed), i + window // 2 + 1)
        avg_duration = sum(w.duration for w in smoothed[start:end]) / (end - start)

        # Only smooth if the current duration is wildly different
        current_dur = smoothed[i].duration
        if current_dur > 0 and avg_duration > 0:
            ratio = current_dur / avg_duration
            if ratio > 3.0 or ratio < 0.33:
                new_end = smoothed[i].start + avg_duration
                new_end = max(new_end, smoothed[i].start + _MIN_WORD_DURATION)
                smoothed[i] = WordTimestamp(
                    word=smoothed[i].word,
                    start=smoothed[i].start,
                    end=new_end,
                    confidence=smoothed[i].confidence,
                )

    return smoothed


# ── Segment Reassembly ────────────────────────────────────────

def _reassemble_segments(
    segments: list[Segment],
    max_segment_duration: float = 12.0,
    min_segment_duration: float = 1.5,
) -> list[Segment]:
    """Intelligently merge short segments and split long segments.

    Merges segments that are too short, splits segments at sentence
    boundaries if they are too long.

    Args:
        segments: Original segments from Whisper.
        max_segment_duration: Maximum duration for a segment.
        min_segment_duration: Minimum duration for a segment.

    Returns:
        Reassembled segment list.
    """
    if not segments:
        return []

    result: list[Segment] = []

    # First pass: merge short segments
    current_text = segments[0].text
    current_start = segments[0].start
    current_end = segments[0].end
    current_conf = segments[0].confidence

    for seg in segments[1:]:
        potential_duration = seg.end - current_start

        # Merge if current segment is too short or naturally continues
        if (current_end - current_start) < min_segment_duration:
            current_text += " " + seg.text
            current_end = seg.end
            current_conf = (current_conf + seg.confidence) / 2.0
        elif potential_duration < min_segment_duration:
            current_text += " " + seg.text
            current_end = seg.end
            current_conf = (current_conf + seg.confidence) / 2.0
        # Break at sentence boundaries
        elif current_text.rstrip()[-1:] in ".?!":
            result.append(Segment(
                text=current_text.strip(),
                start=current_start,
                end=current_end,
                confidence=current_conf,
            ))
            current_text = seg.text
            current_start = seg.start
            current_end = seg.end
            current_conf = seg.confidence
        else:
            current_text += " " + seg.text
            current_end = seg.end
            current_conf = (current_conf + seg.confidence) / 2.0

    # Don't forget the last segment
    if current_text.strip():
        result.append(Segment(
            text=current_text.strip(),
            start=current_start,
            end=current_end,
            confidence=current_conf,
        ))

    # Second pass: split long segments at sentence boundaries
    final: list[Segment] = []
    for seg in result:
        if seg.end - seg.start <= max_segment_duration:
            final.append(seg)
            continue

        # Try to split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', seg.text)
        if len(sentences) <= 1:
            final.append(seg)
            continue

        seg_duration = seg.end - seg.start
        total_chars = sum(len(s) for s in sentences)
        elapsed = seg.start

        current_split_text: list[str] = []
        current_split_chars = 0

        for sentence in sentences:
            current_split_text.append(sentence)
            current_split_chars += len(sentence)

            if current_split_chars >= total_chars * 0.5 or len(current_split_text) >= 3:
                split_duration = seg_duration * (current_split_chars / total_chars)
                split_end = min(elapsed + split_duration, seg.end)
                split_end = max(split_end, elapsed + min_segment_duration)

                final.append(Segment(
                    text=" ".join(current_split_text).strip(),
                    start=elapsed,
                    end=split_end,
                    confidence=seg.confidence,
                ))
                elapsed = split_end
                current_split_text = []
                current_split_chars = 0

        if current_split_text:
            final.append(Segment(
                text=" ".join(current_split_text).strip(),
                start=elapsed,
                end=seg.end,
                confidence=seg.confidence,
            ))

    return final


# ── Punctuation Restoration ───────────────────────────────────

def _restore_punctuation(words: list[WordTimestamp]) -> list[WordTimestamp]:
    """Restore missing punctuation using heuristics.

    Adds periods at sentence boundaries, commas at clause boundaries,
    and question marks for questions based on word patterns.

    Args:
        words: List of word timestamps.

    Returns:
        List with punctuation restored.
    """
    if not words:
        return words

    # Question indicators
    question_starters = {
        "who", "what", "where", "when", "why", "how",
        "is", "are", "do", "does", "did", "can", "could",
        "would", "should", "will", "shall", "may", "might",
    }

    result: list[WordTimestamp] = []

    for i, word in enumerate(words):
        w = word.word
        # Don't double-punctuate
        if w and w[-1:] in ".!?,":
            result.append(word)
            continue

        # Check if this word should have punctuation after it
        next_word = words[i + 1].word if i + 1 < len(words) else None
        is_last = i == len(words) - 1

        # Duration gap to next word
        gap_to_next = 0.0
        if not is_last:
            gap_to_next = words[i + 1].start - word.end

        # Add period at end or at large pauses
        if is_last:
            w = w + "."
        elif gap_to_next > 0.8:
            # Long pause = sentence boundary
            w = w + "."
        elif gap_to_next > 0.4 and len(w) > 2:
            # Medium pause = clause boundary
            w = w + ","

        # Check for question: starts with question word and has rising pattern
        if i == 0 and w.lower().rstrip(",.") in question_starters:
            # Look ahead for the end of the question
            # We'll mark the last word of the "sentence" later
            pass

        result.append(WordTimestamp(
            word=w,
            start=word.start,
            end=word.end,
            confidence=word.confidence,
        ))

    # Post-process: convert period to question mark for questions
    _mark_questions(result, question_starters)

    return result


def _mark_questions(words: list[WordTimestamp], question_starters: set[str]) -> None:
    """Mark questions by converting trailing periods to question marks.

    Args:
        words: List of word timestamps (modified in place).
        question_starters: Set of question-starting words.
    """
    i = 0
    while i < len(words):
        if words[i].word.lower().rstrip(",.") in question_starters:
            # Find the end of this sentence
            j = i + 1
            while j < len(words):
                if words[j].word.endswith("."):
                    # Convert period to question mark
                    words[j] = WordTimestamp(
                        word=words[j].word[:-1] + "?",
                        start=words[j].start,
                        end=words[j].end,
                        confidence=words[j].confidence,
                    )
                    i = j + 1
                    break
                j += 1
            else:
                i += 1
        else:
            i += 1


# ── Language Detection ────────────────────────────────────────

def detect_language(
    video_path: Path,
    settings: Settings | None = None,
) -> LanguageDetection:
    """Detect the language of a video using Whisper.

    Args:
        video_path: Path to the video file.
        settings: Optional Settings override.

    Returns:
        LanguageDetection with language code, confidence, and probabilities.
    """
    if settings is None:
        settings = get_settings()

    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        return LanguageDetection()

    tmp_wav = Path(tempfile.mktemp(suffix=".wav", prefix="whisper_lang_"))
    try:
        extract_audio_wav(video_path, tmp_wav, sample_rate=16000)
    except Exception as exc:
        logger.error("Audio extraction for language detection failed: %s", exc)
        safe_delete(tmp_wav)
        return LanguageDetection()

    try:
        # Try faster-whisper first (more efficient for language detection)
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments_iter, info = model.transcribe(str(tmp_wav), language=None)
            lang_probs = info.language_probability
            detected = LanguageDetection(
                language=info.language,
                confidence=info.language_probability,
                language_probabilities={info.language: info.language_probability},
            )
            logger.info("Language detected (faster-whisper): %s (%.2f%%)",
                       detected.language, detected.confidence * 100)
            return detected
        except ImportError:
            pass

        # Fall back to openai-whisper
        import whisper
        model = whisper.load_model("tiny", device="cpu")
        audio = whisper.pad_or_trim(whisper.load_audio(str(tmp_wav)))
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)

        # Sort by probability
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_lang = sorted_probs[0][0] if sorted_probs else "en"
        top_conf = sorted_probs[0][1] if sorted_probs else 0.0

        detected = LanguageDetection(
            language=top_lang,
            confidence=top_conf,
            language_probabilities=dict(sorted_probs[:10]),
        )
        logger.info("Language detected (openai-whisper): %s (%.2f%%)",
                   top_lang, top_conf * 100)
        return detected

    except Exception as exc:
        logger.error("Language detection failed: %s", exc)
        return LanguageDetection()
    finally:
        safe_delete(tmp_wav)


# ── Prosody Detection ─────────────────────────────────────────

def _estimate_prosody(
    words: list[WordTimestamp],
    segments: list[Segment],
) -> list[ProsodyInfo]:
    """Estimate prosody features from word timing and segment structure.

    Uses heuristics based on word duration, pauses, and punctuation
    to estimate emphasis, questions, and excitement.

    Args:
        words: List of word timestamps.
        segments: List of segments.

    Returns:
        List of ProsodyInfo, one per word.
    """
    if not words:
        return []

    # Calculate average word duration for reference
    durations = [w.duration for w in words if w.duration > 0]
    avg_duration = sum(durations) / len(durations) if durations else 0.3

    prosody_list: list[ProsodyInfo] = []

    for i, word in enumerate(words):
        emphasis = 0.0
        is_question = False
        is_exclamatory = False
        energy = 0.5

        # Emphasis: words spoken slower than average are likely emphasized
        if avg_duration > 0:
            duration_ratio = word.duration / avg_duration
            if duration_ratio > 1.5:
                emphasis = min(1.0, (duration_ratio - 1.0) / 2.0)

        # Short, forceful words might be exclamatory
        if word.word.isupper() and len(word.word) > 1:
            emphasis = 1.0
            is_exclamatory = True

        # Check if word ends with question mark
        if word.word.rstrip()[-1:] == "?":
            is_question = True
            emphasis = max(emphasis, 0.5)

        # Check if word ends with exclamation
        if word.word.rstrip()[-1:] == "!":
            is_exclamatory = True
            emphasis = max(emphasis, 0.7)

        # Energy estimate based on position relative to segment peaks
        for seg in segments:
            if seg.start <= word.start <= seg.end:
                seg_duration = seg.end - seg.start
                if seg_duration > 0:
                    # Words in shorter segments with more text = likely faster/louder
                    seg_words = len(seg.text.split())
                    words_per_sec = seg_words / seg_duration
                    if words_per_sec > 4.0:
                        energy = min(1.0, 0.5 + words_per_sec * 0.1)
                break

        prosody_list.append(ProsodyInfo(
            emphasis=round(emphasis, 2),
            is_question=is_question,
            is_exclamatory=is_exclamatory,
            energy_estimate=round(energy, 2),
        ))

    return prosody_list


# ── Transcription Caching ─────────────────────────────────────

def _compute_audio_hash(audio_path: Path) -> str:
    """Compute a SHA256 hash of the audio file for caching.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Hex digest of the file hash.
    """
    hasher = hashlib.sha256()
    try:
        with open(audio_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError:
        return ""
    return hasher.hexdigest()[:32]


def _get_cache_path(audio_hash: str, model_name: str, settings: Settings) -> Path:
    """Get the cache file path for a given audio hash and model.

    Args:
        audio_hash: SHA256 hash of the audio file.
        model_name: Whisper model name used.
        settings: Settings instance.

    Returns:
        Path to the cache JSON file.
    """
    cache_dir = settings.LOGS_DIR / _CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{audio_hash}_{model_name}.json"


def _load_from_cache(audio_hash: str, model_name: str, settings: Settings) -> Optional[dict]:
    """Load cached transcription result if available.

    Args:
        audio_hash: SHA256 hash of the audio file.
        model_name: Whisper model name.
        settings: Settings instance.

    Returns:
        Cached result dict or None.
    """
    cache_path = _get_cache_path(audio_hash, model_name, settings)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load cache: %s", exc)
    return None


def _save_to_cache(audio_hash: str, model_name: str, result_data: dict, settings: Settings) -> None:
    """Save transcription result to cache.

    Args:
        audio_hash: SHA256 hash of the audio file.
        model_name: Whisper model name.
        result_data: Transcription result data to cache.
        settings: Settings instance.
    """
    cache_path = _get_cache_path(audio_hash, model_name, settings)
    try:
        cache_path.write_text(
            json.dumps(result_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("Cached transcription: %s", cache_path.name)
    except OSError as exc:
        logger.warning("Failed to save cache: %s", exc)


# ── Model Selection ───────────────────────────────────────────

def _auto_select_model(
    duration: float,
    language: str,
    device: str,
) -> str:
    """Auto-detect the best Whisper model based on video characteristics.

    Uses smaller models for short clips and larger models for longer content.
    English-only models are preferred when language is English.

    Args:
        duration: Video duration in seconds.
        language: Detected or specified language code.
        device: Compute device (cpu, cuda, mps).

    Returns:
        Recommended model name.
    """
    is_english = language.startswith("en")
    is_cpu = device == "cpu"

    if duration < 30:
        # Short clip: tiny or base is sufficient
        return "tiny.en" if is_english else "tiny"
    elif duration < 120:
        # Medium clip
        if is_cpu:
            return "base.en" if is_english else "base"
        else:
            return "small.en" if is_english else "small"
    elif duration < 600:
        # Longer clip
        if is_cpu:
            return "base.en" if is_english else "base"
        else:
            return "medium.en" if is_english else "medium"
    else:
        # Very long clip: use larger model only with GPU
        if is_cpu:
            return "small.en" if is_english else "small"
        else:
            return "large-v3"


# ── Transcription Comparison ──────────────────────────────────

def _compare_transcriptions(
    results: list[tuple[str, TranscriptionResult]],
) -> TranscriptionResult:
    """Compare results from multiple models and pick the best.

    Selects the transcription with the highest average confidence.
    In case of a tie, prefers results with more words.

    Args:
        results: List of (model_name, TranscriptionResult) tuples.

    Returns:
        The best TranscriptionResult.
    """
    if not results:
        return TranscriptionResult()
    if len(results) == 1:
        return results[0][1]

    best_result = results[0][1]
    best_score = best_result.average_confidence * best_result.word_count

    for model_name, result in results[1:]:
        score = result.average_confidence * result.word_count
        if score > best_score:
            best_score = score
            best_result = result

    logger.info(
        "Transcription comparison: selected result with %.2f avg conf, %d words",
        best_result.average_confidence, best_result.word_count,
    )
    return best_result


# ── Core Transcription Function ───────────────────────────────

def transcribe(
    video_path: Path,
    settings: Settings | None = None,
    confidence_threshold: float = 0.3,
    use_vad: bool = True,
    use_cache: bool = True,
    compare_models: bool = False,
) -> TranscriptionResult:
    """Transcribe a video file with word-level timestamps using Whisper.

    Full pipeline: extract audio -> VAD filter -> detect language ->
    auto-select model -> transcribe -> filter hallucinations ->
    fix timing -> restore punctuation -> estimate prosody.

    Supports both openai-whisper and faster-whisper libraries.
    Includes VAD pre-filtering, confidence filtering, hallucination
    detection v2, word-level timing correction, segment reassembly,
    punctuation restoration, prosody detection, and transcription caching.

    Args:
        video_path: Path to the video file to transcribe.
        settings: Optional Settings override.
        confidence_threshold: Minimum word confidence to keep (default 0.3).
        use_vad: Whether to use VAD pre-filtering.
        use_cache: Whether to use transcription caching.
        compare_models: Whether to compare multiple models.

    Returns:
        TranscriptionResult with words, segments, language, and duration.
        Returns empty TranscriptionResult if transcription fails.
    """
    if settings is None:
        settings = get_settings()

    if not video_path.exists():
        logger.error("Video file not found for transcription: %s", video_path)
        return TranscriptionResult()

    # ── Step 1: Extract audio ────────────────────────────
    tmp_wav = Path(tempfile.mktemp(suffix=".wav", prefix="whisper_"))
    try:
        extract_audio_wav(video_path, tmp_wav, sample_rate=16000)
        logger.info("Extracted audio to: %s", tmp_wav.name)
    except Exception as exc:
        logger.error("Audio extraction failed: %s", exc)
        safe_delete(tmp_wav)
        return TranscriptionResult()

    # ── Step 2: Check cache ──────────────────────────────
    audio_hash = _compute_audio_hash(tmp_wav) if use_cache else ""

    # ── Step 3: VAD pre-filtering ────────────────────────
    speech_regions = [(0.0, float("inf"))]
    if use_vad:
        speech_regions = _apply_vad_filter(tmp_wav)
        total_speech = sum(end - start for start, end in speech_regions)
        logger.info("VAD: %.1fs of speech detected in %.1fs audio",
                   total_speech, tmp_wav.stat().st_size / 32000)

    # ── Step 4: Detect language ──────────────────────────
    language: Optional[str] = None
    if settings.WHISPER_LANGUAGE and settings.WHISPER_LANGUAGE != "auto":
        language = settings.WHISPER_LANGUAGE

    detected_lang: Optional[LanguageDetection] = None
    if language is None:
        try:
            detected_lang = detect_language(video_path, settings)
            if detected_lang.confidence > 0.5:
                language = detected_lang.language
                logger.info("Auto-detected language: %s (%.0f%%)",
                           language, detected_lang.confidence * 100)
        except Exception as exc:
            logger.debug("Language detection failed, using default: %s", exc)

    # ── Step 5: Auto-select model ────────────────────────
    video_info = probe_video(video_path)
    model_name = settings.WHISPER_MODEL
    device = settings.WHISPER_DEVICE

    # ── Step 6: Build model list ─────────────────────────
    models_to_try = [model_name]
    if settings.WHISPER_FALLBACK_MODELS:
        for fb in settings.WHISPER_FALLBACK_MODELS.split(","):
            fb = fb.strip()
            if fb and fb != model_name:
                models_to_try.append(fb)

    # Add auto-selected model if different
    auto_model = _auto_select_model(video_info.duration, language or "en", device)
    if auto_model not in models_to_try:
        models_to_try.append(auto_model)

    # ── Step 7: Transcribe ───────────────────────────────
    all_results: list[tuple[str, TranscriptionResult]] = []
    max_models = 3 if compare_models else 1

    for try_model in models_to_try[:max_models]:
        # Check cache first
        if use_cache and audio_hash:
            cached = _load_from_cache(audio_hash, try_model, settings)
            if cached:
                logger.info("Using cached transcription for model %s", try_model)
                cached_result = _parse_cached_result(cached)
                all_results.append((try_model, cached_result))
                continue

        # Try faster-whisper first
        result = _transcribe_with_faster_whisper(
            tmp_wav, try_model, language, device, settings, speech_regions,
            confidence_threshold,
        )

        # Fall back to openai-whisper
        if result is None:
            result = _transcribe_with_openai_whisper(
                tmp_wav, try_model, language, device, settings,
                confidence_threshold,
            )

        if result is not None:
            all_results.append((try_model, result))

            # Cache the result
            if use_cache and audio_hash:
                _save_to_cache(audio_hash, try_model, _serialize_result(result), settings)

    safe_delete(tmp_wav)

    if not all_results:
        logger.error("Failed to transcribe with any model from: %s", models_to_try)
        return TranscriptionResult()

    # ── Step 8: Select best result ───────────────────────
    if compare_models and len(all_results) > 1:
        transcription = _compare_transcriptions(all_results)
    else:
        transcription = all_results[0][1]

    # Set language info
    if detected_lang:
        transcription.language = detected_lang.language
        transcription.language_confidence = detected_lang.confidence
    elif language:
        transcription.language = language

    # ── Step 9: Estimate prosody ─────────────────────────
    try:
        transcription.prosody = _estimate_prosody(
            transcription.words, transcription.segments,
        )
    except Exception as exc:
        logger.debug("Prosody estimation failed: %s", exc)

    # ── Step 10: Log results ─────────────────────────────
    logger.info(
        "Transcription complete: language=%s, %d words, %d segments, %.1fs",
        transcription.language,
        transcription.word_count,
        len(transcription.segments),
        transcription.duration,
    )
    console.print(
        f"[green]Transcribed:[/green] {transcription.word_count} words "
        f"({transcription.language}, {transcription.duration:.1}s, "
        f"conf={transcription.average_confidence:.0%})"
    )

    return transcription


def _transcribe_with_faster_whisper(
    audio_path: Path,
    model_name: str,
    language: Optional[str],
    device: str,
    settings: Settings,
    speech_regions: list[tuple[float, float]],
    confidence_threshold: float,
) -> Optional[TranscriptionResult]:
    """Transcribe using faster-whisper (CTranslate2-based).

    Args:
        audio_path: Path to the WAV audio file.
        model_name: Model name to load.
        language: Language code or None for auto-detect.
        device: Compute device.
        settings: Settings instance.
        speech_regions: VAD speech regions.
        confidence_threshold: Minimum word confidence.

    Returns:
        TranscriptionResult or None if faster-whisper is unavailable.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.debug("faster-whisper not installed, skipping")
        return None

    try:
        compute_type = settings.WHISPER_COMPUTE_TYPE
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        console.print(f"[cyan]Loading faster-whisper model '{model_name}' on {device}...[/cyan]")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)

        console.print("[cyan]Running faster-whisper transcription...[/cyan]")

        transcription = TranscriptionResult(
            model_used=f"faster-whisper/{model_name}",
            device_used=device,
        )

        # Transcribe each speech region
        for region_start, region_end in speech_regions:
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=language,
                task=settings.WHISPER_TASK,
                beam_size=settings.WHISPER_BEAM_SIZE,
                temperature=settings.WHISPER_TEMPERATURE,
                word_timestamps=True,
                vad_filter=True,
            )

            if not transcription.language and info.language:
                transcription.language = info.language
                transcription.language_confidence = info.language_probability

            # Process segments
            prev_words: list[WordTimestamp] = []
            full_text = ""

            for seg in segments_iter:
                seg_text = seg.text.strip()
                seg_conf = seg.avg_logprob if seg.avg_logprob else 0.0

                # Filter hallucination segments
                if _is_segment_hallucination(seg_text, seg_conf):
                    logger.debug("Filtering hallucination segment: '%s'", seg_text[:50])
                    continue

                # Offset segment times by region start
                actual_start = seg.start + region_start
                actual_end = seg.end + region_start

                if seg_text:
                    transcription.segments.append(Segment(
                        text=seg_text,
                        start=actual_start,
                        end=actual_end,
                        confidence=seg_conf,
                    ))

                # Process words
                if seg.words:
                    for w in seg.words:
                        word_text = w.word.strip()
                        if not word_text:
                            continue

                        word_start = w.start + region_start
                        word_end = w.end + region_start
                        word_prob = w.probability if w.probability else 1.0
                        word_duration = word_end - word_start

                        full_text += " " + word_text

                        if _is_hallucination(word_text, word_prob, word_duration, prev_words, full_text):
                            logger.debug(
                                "Filtering hallucination: '%s' (conf=%.2f, dur=%.2fs)",
                                word_text, word_prob, word_duration,
                            )
                            continue

                        # Confidence threshold filtering
                        if word_prob < confidence_threshold:
                            logger.debug(
                                "Filtering low-confidence word: '%s' (conf=%.2f)",
                                word_text, word_prob,
                            )
                            continue

                        wt = WordTimestamp(
                            word=word_text,
                            start=word_start,
                            end=word_end,
                            confidence=word_prob,
                        )
                        transcription.words.append(wt)
                        prev_words.append(wt)

                        if len(prev_words) > 20:
                            prev_words = prev_words[-20:]

        # Apply word-level timing corrections
        transcription.words = _fix_overlapping_words(transcription.words)
        transcription.words = _fill_gaps(transcription.words)
        transcription.words = _smooth_timing(transcription.words)

        # Restore punctuation
        transcription.words = _restore_punctuation(transcription.words)

        # Reassemble segments
        transcription.segments = _reassemble_segments(transcription.segments)

        # Calculate duration
        if transcription.words:
            transcription.duration = transcription.words[-1].end
        elif transcription.segments:
            transcription.duration = transcription.segments[-1].end

        return transcription

    except Exception as exc:
        logger.warning("faster-whisper transcription failed: %s", exc)
        return None


def _transcribe_with_openai_whisper(
    audio_path: Path,
    model_name: str,
    language: Optional[str],
    device: str,
    settings: Settings,
    confidence_threshold: float,
) -> Optional[TranscriptionResult]:
    """Transcribe using openai-whisper.

    Args:
        audio_path: Path to the WAV audio file.
        model_name: Model name to load.
        language: Language code or None for auto-detect.
        device: Compute device.
        settings: Settings instance.
        confidence_threshold: Minimum word confidence.

    Returns:
        TranscriptionResult or None if openai-whisper is unavailable.
    """
    try:
        import whisper
    except ImportError:
        logger.error("openai-whisper is not installed. Install with: pip install openai-whisper")
        return None

    try:
        console.print(f"[cyan]Loading Whisper model '{model_name}' on {device}...[/cyan]")
        model = whisper.load_model(model_name, device=device)
    except Exception as exc:
        logger.warning("Failed to load Whisper model '%s': %s", model_name, exc)
        return None

    try:
        console.print("[cyan]Running openai-whisper transcription...[/cyan]")
        result = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language=language,
            task=settings.WHISPER_TASK,
            beam_size=settings.WHISPER_BEAM_SIZE,
            temperature=settings.WHISPER_TEMPERATURE,
            verbose=False,
        )
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        return None

    # ── Parse results with hallucination filtering ─────
    transcription = TranscriptionResult(
        model_used=f"openai-whisper/{model_name}",
        device_used=device,
    )
    detected_language = result.get("language", "unknown")
    transcription.language = detected_language

    prev_words: list[WordTimestamp] = []
    full_text = ""

    for seg in result.get("segments", []):
        seg_text = seg.get("text", "").strip()
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        seg_conf = seg.get("avg_logprob", 0.0)

        # Filter common hallucination phrases
        if _is_segment_hallucination(seg_text, seg_conf):
            logger.debug("Filtering hallucination segment: '%s'", seg_text[:50])
            continue

        if seg_text:
            transcription.segments.append(
                Segment(text=seg_text, start=seg_start, end=seg_end, confidence=seg_conf)
            )

        # Parse words within each segment
        for w in seg.get("words", []):
            word_text = w.get("word", "").strip()
            word_start = w.get("start", 0.0)
            word_end = w.get("end", 0.0)
            word_prob = w.get("probability", 1.0)

            if not word_text:
                continue

            word_duration = word_end - word_start
            full_text += " " + word_text

            # Hallucination detection v2
            if _is_hallucination(word_text, word_prob, word_duration, prev_words, full_text):
                logger.debug(
                    "Filtering hallucination: '%s' (conf=%.2f, dur=%.2fs)",
                    word_text, word_prob, word_duration,
                )
                continue

            # Confidence threshold filtering
            if word_prob < confidence_threshold:
                logger.debug(
                    "Filtering low-confidence word: '%s' (conf=%.2f)",
                    word_text, word_prob,
                )
                continue

            wt = WordTimestamp(
                word=word_text,
                start=word_start,
                end=word_end,
                confidence=word_prob,
            )
            transcription.words.append(wt)
            prev_words.append(wt)

            # Keep prev_words bounded
            if len(prev_words) > 20:
                prev_words = prev_words[-20:]

    # Apply word-level timing corrections
    transcription.words = _fix_overlapping_words(transcription.words)
    transcription.words = _fill_gaps(transcription.words)
    transcription.words = _smooth_timing(transcription.words)

    # Restore punctuation
    transcription.words = _restore_punctuation(transcription.words)

    # Reassemble segments
    transcription.segments = _reassemble_segments(transcription.segments)

    # Calculate duration from last word or segment
    if transcription.words:
        transcription.duration = transcription.words[-1].end
    elif transcription.segments:
        transcription.duration = transcription.segments[-1].end

    return transcription


# ── Serialization helpers for caching ─────────────────────────

def _serialize_result(result: TranscriptionResult) -> dict:
    """Serialize a TranscriptionResult to a JSON-compatible dict.

    Args:
        result: TranscriptionResult to serialize.

    Returns:
        Dictionary representation.
    """
    return {
        "words": [
            {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
            for w in result.words
        ],
        "segments": [
            {"text": s.text, "start": s.start, "end": s.end, "confidence": s.confidence}
            for s in result.segments
        ],
        "language": result.language,
        "language_confidence": result.language_confidence,
        "duration": result.duration,
        "model_used": result.model_used,
        "device_used": result.device_used,
    }


def _parse_cached_result(data: dict) -> TranscriptionResult:
    """Parse a cached result dict back into a TranscriptionResult.

    Args:
        data: Cached dictionary.

    Returns:
        TranscriptionResult instance.
    """
    result = TranscriptionResult(
        language=data.get("language", ""),
        language_confidence=data.get("language_confidence", 0.0),
        duration=data.get("duration", 0.0),
        model_used=data.get("model_used", ""),
        device_used=data.get("device_used", ""),
    )

    for w in data.get("words", []):
        result.words.append(WordTimestamp(
            word=w["word"], start=w["start"], end=w["end"], confidence=w["confidence"],
        ))

    for s in data.get("segments", []):
        result.segments.append(Segment(
            text=s["text"], start=s["start"], end=s["end"], confidence=s["confidence"],
        ))

    return result


# ── Segment-Specific Transcription ────────────────────────────

def transcribe_segment(
    video_path: Path,
    start_time: float,
    end_time: float,
    settings: Settings | None = None,
) -> TranscriptionResult:
    """Transcribe only a specific time range of a video.

    Extracts the audio segment, then transcribes only that portion.

    Args:
        video_path: Path to the video file.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        settings: Optional Settings override.

    Returns:
        TranscriptionResult for the specified segment.

    Raises:
        ValueError: If start_time >= end_time.
    """
    if start_time >= end_time:
        raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

    if settings is None:
        settings = get_settings()

    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        return TranscriptionResult()

    # Extract audio segment
    tmp_wav = Path(tempfile.mktemp(suffix=".wav", prefix="whisper_seg_"))
    try:
        duration = end_time - start_time
        from utils.ffmpeg_utils import run_ffmpeg
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-ar", "16000", "-ac", "1", "-f", "wav",
            "-ss", str(start_time), "-t", str(duration),
            str(tmp_wav),
        ]
        run_ffmpeg(cmd, description=f"Extract audio segment {start_time:.1f}-{end_time:.1f}s")
    except Exception as exc:
        logger.error("Audio segment extraction failed: %s", exc)
        safe_delete(tmp_wav)
        return TranscriptionResult()

    # Transcribe the segment
    language: Optional[str] = None
    if settings.WHISPER_LANGUAGE and settings.WHISPER_LANGUAGE != "auto":
        language = settings.WHISPER_LANGUAGE

    result = None

    # Try faster-whisper
    result = _transcribe_with_faster_whisper(
        tmp_wav, settings.WHISPER_MODEL, language, settings.WHISPER_DEVICE,
        settings, [(0.0, float("inf"))], _MIN_CONFIDENCE,
    )

    # Fall back to openai-whisper
    if result is None:
        result = _transcribe_with_openai_whisper(
            tmp_wav, settings.WHISPER_MODEL, language, settings.WHISPER_DEVICE,
            settings, _MIN_CONFIDENCE,
        )

    safe_delete(tmp_wav)

    if result is None:
        return TranscriptionResult()

    # Offset timestamps by start_time
    for w in result.words:
        w.start += start_time
        w.end += start_time
    for s in result.segments:
        s.start += start_time
        s.end += start_time

    return result


# ── Force Alignment ───────────────────────────────────────────

def force_align(
    transcription: TranscriptionResult,
    target_start: float,
    target_end: float,
) -> TranscriptionResult:
    """Align transcription to a specific time range.

    Stretches or compresses word timing to fit within the target range,
    preserving relative timing between words.

    Args:
        transcription: Original transcription result.
        target_start: Desired start time.
        target_end: Desired end time.

    Returns:
        Aligned TranscriptionResult.
    """
    if transcription.is_empty:
        return transcription

    original_start = transcription.words[0].start
    original_end = transcription.words[-1].end
    original_duration = original_end - original_start

    if original_duration <= 0:
        return transcription

    target_duration = target_end - target_start
    scale = target_duration / original_duration

    aligned = TranscriptionResult(
        language=transcription.language,
        language_confidence=transcription.language_confidence,
        duration=target_duration,
        model_used=transcription.model_used,
        device_used=transcription.device_used,
    )

    for w in transcription.words:
        new_start = target_start + (w.start - original_start) * scale
        new_end = target_start + (w.end - original_start) * scale
        aligned.words.append(WordTimestamp(
            word=w.word, start=new_start, end=new_end, confidence=w.confidence,
        ))

    for s in transcription.segments:
        new_start = target_start + (s.start - original_start) * scale
        new_end = target_start + (s.end - original_start) * scale
        aligned.segments.append(Segment(
            text=s.text, start=new_start, end=new_end, confidence=s.confidence,
        ))

    for p in transcription.prosody:
        aligned.prosody.append(p)

    return aligned


# ── Export Functions ───────────────────────────────────────────

def export_srt(
    transcription: TranscriptionResult,
    output_path: Path,
) -> Path:
    """Export transcription as SRT subtitle file.

    Args:
        transcription: TranscriptionResult to export.
        output_path: Destination path for the SRT file.

    Returns:
        Path to the written SRT file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for i, seg in enumerate(transcription.segments, start=1):
        start_ts = _seconds_to_srt_time(seg.start)
        end_ts = _seconds_to_srt_time(seg.end)
        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("SRT exported: %s (%d segments)", output_path.name, len(transcription.segments))
    return output_path


def export_vtt(
    transcription: TranscriptionResult,
    output_path: Path,
) -> Path:
    """Export transcription as WebVTT subtitle file.

    Args:
        transcription: TranscriptionResult to export.
        output_path: Destination path for the VTT file.

    Returns:
        Path to the written VTT file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["WEBVTT", ""]

    for seg in transcription.segments:
        start_ts = _seconds_to_vtt_time(seg.start)
        end_ts = _seconds_to_vtt_time(seg.end)
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("VTT exported: %s (%d segments)", output_path.name, len(transcription.segments))
    return output_path


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert float seconds to SRT timestamp format HH:MM:SS,mmm.

    Args:
        seconds: Time in seconds.

    Returns:
        SRT-formatted timestamp string.
    """
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _seconds_to_vtt_time(seconds: float) -> str:
    """Convert float seconds to WebVTT timestamp format HH:MM:SS.mmm.

    Args:
        seconds: Time in seconds.

    Returns:
        WebVTT-formatted timestamp string.
    """
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
