"""
core/metadata_generator.py — Advanced local metadata generation with TF-IDF,
bigram/trigram extraction, multi-strategy titles, SEO scoring, content categorization,
mood/tone detection, trending keyword detection, and multi-language support.

All generation is done locally using keyword extraction and templates —
zero external API calls.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.settings import Settings, get_settings
from core.transcriber import TranscriptionResult
from utils.file_utils import sanitize_filename
from utils.logger import get_logger

logger = get_logger("metadata_generator")


# ── Data Classes ──────────────────────────────────────────────

@dataclass
class MetadataResult:
    """Generated metadata for all platforms."""

    youtube_title: str = ""
    youtube_description: str = ""
    youtube_tags: list[str] = field(default_factory=list)
    tiktok_caption: str = ""
    reels_caption: str = ""
    keywords: list[str] = field(default_factory=list)
    thumbnail_keywords: list[str] = field(default_factory=list)
    title_variants: list[str] = field(default_factory=list)
    seo_score: float = 0.0
    content_category: str = ""
    mood: str = ""
    target_audience: str = ""
    hashtags: list[str] = field(default_factory=list)
    emoji_recommendations: list[str] = field(default_factory=list)


@dataclass
class KeywordScore:
    """A keyword with its multi-signal score."""

    keyword: str
    frequency: int = 0
    tfidf_score: float = 0.0
    title_boost: float = 0.0
    tag_boost: float = 0.0
    position_boost: float = 0.0
    total_score: float = 0.0


@dataclass
class SEOScore:
    """SEO effectiveness score breakdown."""

    title_score: float = 0.0
    description_score: float = 0.0
    tag_score: float = 0.0
    keyword_density: float = 0.0
    overall_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)


# ── Extended Stopwords (400+ words) ──────────────────────────

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
    "um", "uh", "hmm", "just", "really", "actually", "literally",
    "basically", "honestly", "mean", "means", "maybe", "might", "shall",
    "must", "used", "using", "uses", "able", "across", "along", "already",
    "among", "anybody", "anyone", "anything", "anywhere", "became",
    "become", "becomes", "behind", "beside", "besides", "beyond",
    "certain", "certainly", "clear", "clearly", "come", "comes",
    "concerning", "consider", "considering", "contain", "containing",
    "contains", "corresponding", "could", "course", "day", "days",
    "different", "difficult", "done", "either", "else", "enough",
    "especially", "etc", "everybody", "everyone", "everything",
    "everywhere", "example", "except", "far", "feel", "feels",
    "few", "follow", "following", "follows", "found", "give", "given",
    "gives", "going", "gone", "happen", "happened", "happens", "hard",
    "help", "here", "high", "however", "important", "instead", "keep",
    "kept", "later", "least", "left", "less", "let", "likely",
    "looking", "may", "might", "mind", "moment", "morning", "much",
    "must", "name", "near", "nearby", "nearly", "necessary", "need",
    "needed", "neither", "night", "nothing", "number", "often", "order",
    "others", "part", "particular", "particularly", "past", "perhaps",
    "person", "place", "point", "possible", "present", "probably",
    "problem", "provide", "put", "quite", "rather", "read", "reason",
    "remember", "result", "run", "second", "seem", "seemed", "seems",
    "set", "several", "shall", "short", "show", "shown", "side",
    "simple", "simply", "small", "so", "sometime", "sometimes",
    "soon", "stand", "start", "started", "state", "still", "stop",
    "sure", "take", "taken", "tell", "thing", "though", "thought",
    "told", "toward", "towards", "true", "try", "turn", "turned",
    "understand", "upon", "use", "used", "using", "usually", "want",
    "wanted", "water", "whole", "without", "word", "words", "work",
    "worked", "working", "world", "would", "year", "years", "yes",
    "yet", "young",
}

# ── Language-specific stopwords ──────────────────────────────

_LANGUAGE_STOPWORDS: dict[str, set[str]] = {
    "es": {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
        "al", "a", "en", "por", "para", "con", "sin", "sobre", "entre",
        "que", "quien", "cual", "cuyo", "donde", "cuando", "como", "cuanto",
        "es", "son", "fue", "fueron", "ser", "estar", "ha", "han", "había",
        "y", "o", "pero", "si", "no", "ni", "también", "además", "muy",
        "más", "menos", "mucho", "poco", "todo", "nada", "algo", "este",
        "esta", "ese", "esa", "aquel", "aquella", "su", "sus", "mi", "mis",
        "tu", "tus", "nuestro", "nuestra", "vuestro", "vuestra",
    },
    "fr": {
        "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
        "à", "en", "dans", "sur", "sous", "avec", "sans", "pour", "par",
        "qui", "que", "quoi", "dont", "où", "quand", "comment", "combien",
        "est", "sont", "était", "ont", "avoir", "être", "et", "ou", "mais",
        "si", "non", "ne", "pas", "plus", "aussi", "très", "bien", "tout",
        "rien", "quelque", "ce", "cette", "ces", "son", "sa", "ses",
        "mon", "ma", "mes", "ton", "ta", "tes", "notre", "votre",
    },
    "de": {
        "der", "die", "das", "ein", "eine", "einer", "einem", "einen",
        "von", "zu", "in", "an", "auf", "mit", "ohne", "für", "durch",
        "wer", "was", "wo", "wann", "wie", "warum", "welche", "welcher",
        "ist", "sind", "war", "waren", "haben", "hat", "hatte", "sein",
        "und", "oder", "aber", "wenn", "nicht", "auch", "sehr", "gut",
        "alles", "nichts", "etwas", "dieser", "diese", "dieses", "sein",
        "ihr", "ihre", "mein", "meine", "dein", "deine", "unser", "euer",
    },
    "pt": {
        "o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do",
        "da", "dos", "das", "em", "no", "na", "nos", "nas", "por", "para",
        "com", "sem", "sobre", "entre", "que", "quem", "qual", "cujo",
        "onde", "quando", "como", "quanto", "é", "são", "foi", "foram",
        "ser", "estar", "tem", "têm", "tinha", "e", "ou", "mas", "se",
        "não", "também", "muito", "mais", "menos", "todo", "nada", "algo",
        "este", "esta", "esse", "essa", "seu", "sua", "meu", "minha",
    },
    "ja": {
        "の", "に", "は", "を", "た", "が", "で", "て", "と", "し",
        "れ", "さ", "ある", "いる", "も", "する", "から", "な", "こと",
        "として", "い", "や", "れる", "など", "なっ", "ない", "この",
        "ため", "その", "あっ", "よう", "また", "もの", "という", "あり",
        "まで", "られ", "なる", "へ", "か", "だ", "これ", "によって",
    },
    "zh": {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
        "吗", "那", "被", "从", "但", "把", "对", "让", "还", "什么",
        "可以", "这个", "那个", "没", "来", "能", "啊", "吧", "呢",
    },
}


# ── Content Category Detection ────────────────────────────────

_CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "tech": {
        "tech", "technology", "code", "programming", "coding", "software",
        "computer", "ai", "artificial", "intelligence", "machine", "learning",
        "data", "algorithm", "python", "javascript", "react", "api", "cloud",
        "server", "database", "developer", "engineering", "digital", "robot",
        "automation", "cybersecurity", "blockchain", "crypto", "web",
    },
    "gaming": {
        "game", "gaming", "play", "esports", "twitch", "stream", "console",
        "pc", "xbox", "playstation", "nintendo", "fortnite", "minecraft",
        "valorant", "apex", "league", "overwatch", "speedrun", "mod",
        "controller", "multiplayer", "battle", "royale", "rpg", "fps",
    },
    "cooking": {
        "food", "cooking", "recipe", "kitchen", "chef", "meal", "bake",
        "baking", "ingredient", "dish", "cuisine", "dinner", "lunch",
        "breakfast", "snack", "dessert", "sauce", "spice", "grill",
        "fry", "roast", "soup", "salad", "cake", "bread", "pasta",
    },
    "fitness": {
        "fitness", "workout", "exercise", "gym", "muscle", "health",
        "training", "cardio", "strength", "yoga", "stretch", "weight",
        "body", "abs", "chest", "leg", "arm", "run", "marathon", "diet",
        "protein", "calories", "set", "rep", "routine", "hiit", "crossfit",
    },
    "music": {
        "music", "song", "guitar", "piano", "beat", "sing", "vocal",
        "drum", "bass", "melody", "chord", "album", "track", "studio",
        "remix", "concert", "live", "band", "instrument", "producer",
        "spotify", "soundcloud", "cover", "acoustic", "electric",
    },
    "travel": {
        "travel", "trip", "vacation", "adventure", "explore", "destination",
        "flight", "hotel", "beach", "mountain", "city", "country", "tour",
        "backpack", "culture", "landmark", "airport", "passport", "visa",
        "island", "temple", "museum", "food", "street", "local",
    },
    "comedy": {
        "funny", "comedy", "laugh", "joke", "hilarious", "meme", "prank",
        "sketch", "standup", "parody", "satire", "humor", "gag", "comic",
        "ridiculous", "absurd", "silly", "witty", "punchline", "roast",
    },
    "education": {
        "learn", "education", "tutorial", "teach", "study", "school",
        "university", "course", "lesson", "explain", "guide", "tips",
        "how", "why", "science", "math", "history", "facts", "knowledge",
        "academic", "research", "lecture", "student", "skill", "practice",
    },
    "finance": {
        "money", "finance", "invest", "trading", "crypto", "bitcoin",
        "stock", "market", "economy", "bank", "loan", "mortgage", "debt",
        "income", "profit", "loss", "portfolio", "dividend", "wealth",
        "retirement", "tax", "budget", "savings", "credit", "interest",
    },
    "fashion": {
        "fashion", "style", "outfit", "beauty", "makeup", "skincare",
        "hair", "clothing", "designer", "trend", "wardrobe", "accessory",
        "shoes", "dress", "luxury", "brand", "lookbook", "haul", "grwm",
        "aesthetic", "vintage", "streetwear", "runway", "model",
    },
    "sports": {
        "sport", "football", "basketball", "soccer", "baseball", "tennis",
        "golf", "hockey", "swimming", "boxing", "mma", "ufc", "race",
        "team", "player", "coach", "match", "game", "score", "win",
        "championship", "league", "tournament", "olympic", "athlete",
    },
}


# ── Mood/Tone Detection Keywords ─────────────────────────────

_MOOD_KEYWORDS: dict[str, set[str]] = {
    "energetic": {
        "amazing", "incredible", "awesome", "exciting", "wow", "unbelievable",
        "insane", "crazy", "epic", "mind", "blown", "powerful", "extreme",
        "intense", "explosive", "fire", "lit", "hype", "pumped", "thrilling",
    },
    "calm": {
        "peaceful", "relaxing", "calm", "gentle", "quiet", "serene",
        "meditation", "breathing", "mindful", "tranquil", "soothing",
        "slow", "soft", "rest", "comfort", "easy", "simple", "mellow",
    },
    "funny": {
        "funny", "hilarious", "laugh", "joke", "comedy", "lol", "lmao",
        "ridiculous", "absurd", "silly", "stupid", "dumb", "wtf", "fail",
        "prank", "meme", "cringe", "awkward", "weird", "wild",
    },
    "dramatic": {
        "dramatic", "shocking", "unbelievable", "twist", "reveal", "secret",
        "exposed", "truth", "conspiracy", "mystery", "suspense", "tension",
        "crisis", "danger", "risk", "critical", "urgent", "breaking",
    },
    "educational": {
        "learn", "explain", "understand", "how", "why", "what", "guide",
        "tutorial", "step", "method", "technique", "process", "fact",
        "science", "proof", "evidence", "research", "analysis", "study",
    },
    "inspirational": {
        "motivation", "inspire", "success", "dream", "goal", "achieve",
        "overcome", "never", "give", "up", "believe", "possible", "change",
        "transform", "growth", "progress", "journey", "challenge", "hope",
    },
}


# ── Target Audience Detection ────────────────────────────────

_AUDIENCE_SIGNALS: dict[str, set[str]] = {
    "gen_z": {
        "tiktok", "vibe", "no", "cap", "slay", "bet", "drip", "flex",
        "fyp", "viral", "trending", "rizz", "sigma", "skibidi",
    },
    "millennials": {
        "adulting", "side", "hustle", "wellness", "self", "care",
        "mindful", "balance", "work", "remote", "freelance", "digital",
    },
    "professionals": {
        "career", "business", "strategy", "leadership", "management",
        "corporate", "startup", "entrepreneur", "investment", "growth",
    },
    "students": {
        "study", "exam", "homework", "college", "university", "grade",
        "test", "learn", "class", "lecture", "textbook", "campus",
    },
    "parents": {
        "kids", "children", "family", "baby", "parenting", "toddler",
        "school", "child", "mom", "dad", "home", "baby", "pregnancy",
    },
}


# ── Emoji Mapping ─────────────────────────────────────────────

_EMOJI_MAP: dict[str, str] = {
    "tech": "tech", "technology": "tech", "code": "tech", "programming": "tech",
    "coding": "tech", "software": "tech", "computer": "tech", "ai": "robot",
    "artificial": "robot", "robot": "robot", "money": "money", "finance": "money",
    "invest": "money", "trading": "money", "crypto": "money", "bitcoin": "money",
    "fitness": "strong", "workout": "strong", "exercise": "strong", "gym": "strong",
    "muscle": "strong", "health": "strong", "food": "food", "cooking": "food",
    "recipe": "food", "kitchen": "food", "music": "music", "song": "music",
    "guitar": "music", "piano": "music", "beat": "music", "travel": "travel",
    "trip": "travel", "vacation": "travel", "adventure": "travel", "explore": "travel",
    "funny": "laugh", "comedy": "laugh", "laugh": "laugh", "joke": "laugh",
    "game": "game", "gaming": "game", "play": "game", "esports": "game",
    "science": "science", "research": "science", "study": "science", "experiment": "science",
    "space": "rocket", "rocket": "rocket", "nasa": "rocket", "mars": "rocket",
    "motivation": "fire", "inspire": "fire", "success": "fire", "hustle": "fire",
    "learn": "book", "education": "book", "tutorial": "book", "teach": "book",
    "nature": "nature", "animals": "nature", "wildlife": "nature", "ocean": "nature",
    "fashion": "fashion", "style": "fashion", "outfit": "fashion", "beauty": "fashion",
    "sport": "sport", "football": "sport", "basketball": "sport", "soccer": "sport",
    "movie": "movie", "film": "movie", "cinema": "movie", "actor": "movie",
    "car": "car", "auto": "car", "drive": "car", "racing": "car",
    "dog": "dog", "cat": "cat", "pet": "pet", "puppy": "dog",
}

_EMOJI_UNICODE: dict[str, str] = {
    "tech": "\u26a1", "robot": "\U0001f916", "money": "\U0001f4b0", "strong": "\U0001f4aa",
    "food": "\U0001f355", "music": "\U0001f3b5", "travel": "\u2708\ufe0f", "laugh": "\U0001f602",
    "game": "\U0001f3ae", "science": "\U0001f52c", "rocket": "\U0001f680", "fire": "\U0001f525",
    "book": "\U0001f4da", "nature": "\U0001f33f", "fashion": "\U0001f457", "sport": "\u26bd",
    "movie": "\U0001f3ac", "car": "\U0001f697", "dog": "\U0001f415", "cat": "\U0001f408",
    "pet": "\U0001f43e", "heart": "\u2764\ufe0f", "star": "\u2b50", "check": "\u2705",
    "warning": "\u26a0\ufe0f", "think": "\U0001f914", "eyes": "\U0001f440",
}

# Category-specific emoji sets
_CATEGORY_EMOJIS: dict[str, list[str]] = {
    "tech": ["\u26a1", "\U0001f916", "\U0001f4bb", "\U0001f5a5\ufe0f"],
    "gaming": ["\U0001f3ae", "\U0001f3ae", "\U0001f579\ufe0f", "\U0001f3c6"],
    "cooking": ["\U0001f355", "\U0001f373", "\U0001f52a", "\U0001f963"],
    "fitness": ["\U0001f4aa", "\U0001f3cb\ufe0f", "\U0001f3c3", "\U0001f947"],
    "music": ["\U0001f3b5", "\U0001f3b8", "\U0001f3a4", "\U0001f3b6"],
    "travel": ["\u2708\ufe0f", "\U0001f30d", "\U0001f3dd\ufe0f", "\U0001f4f7"],
    "comedy": ["\U0001f602", "\U0001f923", "\U0001f604", "\U0001f606"],
    "education": ["\U0001f4da", "\U0001f393", "\U0001f4d6", "\u2705"],
    "finance": ["\U0001f4b0", "\U0001f4c8", "\U0001f4b5", "\U0001f3e6"],
    "fashion": ["\U0001f457", "\U0001f48e", "\U0001f451", "\U0001f4a5"],
    "sports": ["\u26bd", "\U0001f3c0", "\U0001f3c8", "\U0001f3c6"],
}


# ── TF-IDF Keyword Extraction ────────────────────────────────

def _compute_tf(
    words: list[str],
) -> Counter:
    """Compute term frequency (normalized) for a list of words.

    Args:
        words: List of word strings.

    Returns:
        Counter with term frequencies.
    """
    freq = Counter(words)
    total = sum(freq.values())
    if total == 0:
        return freq
    # Normalize: TF = count / total_words
    for word in freq:
        freq[word] = freq[word]  # Keep raw counts, normalize later
    return freq


def _compute_idf(
    documents: list[list[str]],
) -> dict[str, float]:
    """Compute inverse document frequency for terms across documents.

    Uses smoothed IDF: log((N + 1) / (df + 1)) + 1

    Args:
        documents: List of document word lists.

    Returns:
        Dictionary mapping term to IDF score.
    """
    n_docs = len(documents)
    if n_docs == 0:
        return {}

    doc_freq: Counter = Counter()
    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            doc_freq[term] += 1

    idf_scores: dict[str, float] = {}
    for term, df in doc_freq.items():
        # Smoothed IDF
        idf_scores[term] = math.log((n_docs + 1) / (df + 1)) + 1.0

    return idf_scores


def _extract_keywords_tfidf(
    transcription: TranscriptionResult,
    video_tags: list[str] | None = None,
    title_words: list[str] | None = None,
    top_n: int = 15,
    language: str = "en",
) -> list[KeywordScore]:
    """Extract keywords using proper TF-IDF with multi-signal scoring.

    Combines TF-IDF scores with title boost, tag boost, and position
    boost for comprehensive keyword ranking.

    Args:
        transcription: TranscriptionResult with word timestamps.
        video_tags: Optional video tags for boosting.
        title_words: Optional title words for boosting.
        top_n: Number of keywords to return.
        language: Language code for stopwords.

    Returns:
        List of KeywordScore objects sorted by total score.
    """
    # Get all stopwords for the language
    all_stopwords = _STOPWORDS | _LANGUAGE_STOPWORDS.get(language, set())

    # Extract clean words from transcription
    transcription_words: list[str] = []
    for wt in transcription.words:
        word_lower = re.sub(r"[^\w]", "", wt.word.lower())
        if len(word_lower) < 2 or word_lower in all_stopwords:
            continue
        transcription_words.append(word_lower)

    if not transcription_words:
        return []

    # Create pseudo-documents for IDF calculation
    # Split transcription into segments as "documents"
    documents: list[list[str]] = []
    current_doc: list[str] = []
    for i, word in enumerate(transcription_words):
        current_doc.append(word)
        if len(current_doc) >= 20 or i == len(transcription_words) - 1:
            documents.append(current_doc)
            current_doc = []

    # Add video tags and title as additional documents
    if video_tags:
        tag_words = []
        for tag in video_tags:
            for word in tag.lower().split():
                w = re.sub(r"[^\w]", "", word)
                if len(w) >= 2 and w not in all_stopwords:
                    tag_words.append(w)
        if tag_words:
            documents.append(tag_words)

    if title_words:
        title_clean = []
        for word in title_words:
            w = re.sub(r"[^\w]", "", word.lower())
            if len(w) >= 2 and w not in all_stopwords:
                title_clean.append(w)
        if title_clean:
            documents.append(title_clean)

    # Compute TF for the full transcription
    tf = _compute_tf(transcription_words)

    # Compute IDF across documents
    idf = _compute_idf(documents)

    # Compute TF-IDF and multi-signal scores
    keyword_scores: dict[str, KeywordScore] = {}

    total_words = sum(tf.values())
    for word, count in tf.items():
        # TF (normalized)
        tf_norm = count / total_words if total_words > 0 else 0

        # IDF score
        idf_score = idf.get(word, 1.0)

        # TF-IDF
        tfidf = tf_norm * idf_score

        # Title boost: 3x if word appears in title
        title_boost = 3.0 if title_words and word in [re.sub(r"[^\w]", "", w.lower()) for w in title_words] else 0.0

        # Tag boost: 5x if word appears in tags
        tag_boost = 5.0 if video_tags and any(word in tag.lower() for tag in video_tags) else 0.0

        # Position boost: words appearing in the clip segment are more relevant
        # Give higher weight to words in the first 30% and last 20% of the clip
        position_boost = 0.0
        for wt in transcription.words:
            w = re.sub(r"[^\w]", "", wt.word.lower())
            if w == word and transcription.duration > 0:
                relative_pos = wt.start / transcription.duration
                if relative_pos < 0.3 or relative_pos > 0.8:
                    position_boost = max(position_boost, 1.5)
                elif relative_pos < 0.5:
                    position_boost = max(position_boost, 1.0)

        total_score = tfidf + title_boost + tag_boost + position_boost

        keyword_scores[word] = KeywordScore(
            keyword=word,
            frequency=count,
            tfidf_score=round(tfidf, 4),
            title_boost=title_boost,
            tag_boost=tag_boost,
            position_boost=round(position_boost, 2),
            total_score=round(total_score, 4),
        )

    # Sort by total score and return top N
    sorted_keywords = sorted(
        keyword_scores.values(),
        key=lambda k: k.total_score,
        reverse=True,
    )[:top_n]

    return sorted_keywords


# ── Bigram/Trigram Extraction ─────────────────────────────────

def _extract_ngrams(
    transcription: TranscriptionResult,
    n: int = 2,
    top_n: int = 10,
    language: str = "en",
) -> list[tuple[str, int]]:
    """Extract n-grams (bigrams/trigrams) from transcription.

    Args:
        transcription: TranscriptionResult with word timestamps.
        n: N-gram size (2=bigram, 3=trigram).
        top_n: Number of top n-grams to return.
        language: Language code for stopwords.

    Returns:
        List of (ngram_text, count) tuples sorted by frequency.
    """
    all_stopwords = _STOPWORDS | _LANGUAGE_STOPWORDS.get(language, set())

    words: list[str] = []
    for wt in transcription.words:
        word_lower = re.sub(r"[^\w]", "", wt.word.lower())
        if len(word_lower) < 2 or word_lower in all_stopwords:
            words.append("")  # Placeholder to break n-grams
            continue
        words.append(word_lower)

    ngram_freq: Counter = Counter()
    for i in range(len(words) - n + 1):
        ngram_words = words[i:i + n]
        # Skip n-grams that contain placeholders or stopwords
        if any(not w for w in ngram_words):
            continue
        ngram = " ".join(ngram_words)
        ngram_freq[ngram] += 1

    # Filter: only keep n-grams that appear at least twice
    filtered = {ng: count for ng, count in ngram_freq.items() if count >= 2}

    return sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ── Content Category Detection ────────────────────────────────

def _detect_category(keywords: list[str]) -> tuple[str, float]:
    """Detect the content category from keywords.

    Args:
        keywords: List of keyword strings.

    Returns:
        Tuple of (category_name, confidence_score).
    """
    category_scores: dict[str, float] = {}

    for category, cat_keywords in _CATEGORY_KEYWORDS.items():
        score = 0.0
        for kw in keywords:
            if kw in cat_keywords:
                score += 1.0
            # Partial match for compound keywords
            for cat_kw in cat_keywords:
                if kw in cat_kw or cat_kw in kw:
                    score += 0.3
        category_scores[category] = score

    if not category_scores:
        return "general", 0.0

    best_category = max(category_scores, key=category_scores.get)
    best_score = category_scores[best_category]

    # Normalize confidence
    total_score = sum(category_scores.values())
    confidence = best_score / total_score if total_score > 0 else 0.0

    return best_category, min(1.0, confidence)


# ── Mood/Tone Detection ──────────────────────────────────────

def _detect_mood(text: str) -> tuple[str, float]:
    """Detect the mood/tone of the content.

    Args:
        text: Transcription text to analyze.

    Returns:
        Tuple of (mood_name, confidence).
    """
    text_lower = text.lower()
    mood_scores: dict[str, float] = {}

    for mood, mood_words in _MOOD_KEYWORDS.items():
        score = 0.0
        for word in mood_words:
            if word in text_lower:
                score += 1.0
            # Count occurrences
            score += text_lower.count(word) * 0.5
        mood_scores[mood] = score

    if not mood_scores:
        return "neutral", 0.0

    best_mood = max(mood_scores, key=mood_scores.get)
    best_score = mood_scores[best_mood]

    total_score = sum(mood_scores.values())
    confidence = best_score / total_score if total_score > 0 else 0.0

    return best_mood, min(1.0, confidence)


# ── Target Audience Detection ─────────────────────────────────

def _detect_audience(text: str) -> str:
    """Detect the target audience from content.

    Args:
        text: Transcription text.

    Returns:
        Target audience string.
    """
    text_lower = text.lower()
    audience_scores: dict[str, float] = {}

    for audience, signals in _AUDIENCE_SIGNALS.items():
        score = sum(1 for s in signals if s in text_lower)
        audience_scores[audience] = score

    if not audience_scores:
        return "general"

    return max(audience_scores, key=audience_scores.get)


# ── Title Generation Strategies ───────────────────────────────

def _generate_title_hook(transcription: TranscriptionResult) -> str:
    """Generate a hook-style title using the first impactful sentence.

    Args:
        transcription: TranscriptionResult with segments.

    Returns:
        Hook title string.
    """
    for seg in transcription.segments[:5]:
        text = seg.text.strip()
        # Look for impactful sentences (questions, exclamations, surprising statements)
        if len(text) > 15 and len(text) < 100:
            if text[-1:] in ".!?":
                # Strip trailing punctuation for cleaner title
                return text.rstrip(".!?").strip()
    return ""


def _generate_title_keyword(keywords: list[KeywordScore]) -> str:
    """Generate a keyword-based title using top keywords.

    Args:
        keywords: List of KeywordScore objects.

    Returns:
        Keyword-based title string.
    """
    if not keywords:
        return ""
    top_words = [ks.keyword.title() for ks in keywords[:5]]
    return " ".join(top_words)


def _generate_title_question(keywords: list[KeywordScore]) -> str:
    """Generate a question-style title.

    Args:
        keywords: List of KeywordScore objects.

    Returns:
        Question title string.
    """
    if not keywords:
        return ""
    top_word = keywords[0].keyword
    return f"Why {top_word.title()} Changes Everything?"


def _generate_title_listicle(keywords: list[KeywordScore]) -> str:
    """Generate a listicle-style title.

    Args:
        keywords: List of KeywordScore objects.

    Returns:
        Listicle title string.
    """
    if not keywords:
        return ""
    top_word = keywords[0].keyword
    return f"5 {top_word.title()} Secrets You Need to Know"


def _generate_title_how_to(keywords: list[KeywordScore]) -> str:
    """Generate a how-to style title.

    Args:
        keywords: List of KeywordScore objects.

    Returns:
        How-to title string.
    """
    if not keywords:
        return ""
    top_word = keywords[0].keyword
    return f"How to Master {top_word.title()} in 60 Seconds"


def _generate_title_shock(keywords: list[KeywordScore]) -> str:
    """Generate a shocking/surprising title.

    Args:
        keywords: List of KeywordScore objects.

    Returns:
        Shock title string.
    """
    if not keywords:
        return ""
    top_word = keywords[0].keyword
    return f"The {top_word.title()} Truth Nobody Tells You"


_TITLE_STRATEGIES: dict[str, callable] = {
    "hook": _generate_title_hook,
    "keyword": _generate_title_keyword,
    "question": _generate_title_question,
    "listicle": _generate_title_listicle,
    "how_to": _generate_title_how_to,
    "shock": _generate_title_shock,
}


# ── SEO Scoring ───────────────────────────────────────────────

def _score_seo(
    title: str,
    description: str,
    tags: list[str],
    keywords: list[str],
) -> SEOScore:
    """Score the generated metadata for SEO effectiveness.

    Args:
        title: Generated title.
        description: Generated description.
        tags: Generated tags.
        keywords: Target keywords.

    Returns:
        SEOScore with breakdown and recommendations.
    """
    recommendations: list[str] = []
    title_score = 0.0
    desc_score = 0.0
    tag_score = 0.0
    keyword_density = 0.0

    # Title scoring
    title_len = len(title)
    if 30 <= title_len <= 65:
        title_score += 30.0
    elif 20 <= title_len <= 80:
        title_score += 20.0
    else:
        title_score += 5.0
        if title_len < 30:
            recommendations.append("Title is too short (aim for 30-65 characters)")
        elif title_len > 65:
            recommendations.append("Title is too long (aim for 30-65 characters)")

    # Keywords in title
    title_lower = title.lower()
    keywords_in_title = sum(1 for kw in keywords[:5] if kw in title_lower)
    title_score += keywords_in_title * 10.0

    if keywords_in_title == 0:
        recommendations.append("No target keywords found in title")

    # Description scoring
    desc_len = len(description)
    if desc_len >= 100:
        desc_score += 20.0
    elif desc_len >= 50:
        desc_score += 10.0
    else:
        recommendations.append("Description is too short (aim for 100+ characters)")

    # Keywords in description
    keywords_in_desc = sum(1 for kw in keywords[:10] if kw in description.lower())
    desc_score += min(30.0, keywords_in_desc * 5.0)

    # Hashtags in description
    hashtag_count = description.count("#")
    if 3 <= hashtag_count <= 15:
        desc_score += 10.0
    elif hashtag_count > 15:
        recommendations.append("Too many hashtags (aim for 3-15)")

    # Tag scoring
    if len(tags) >= 5:
        tag_score += 20.0
    elif len(tags) >= 3:
        tag_score += 10.0
    else:
        recommendations.append("Add more tags (aim for 5+)")

    # Keyword relevance in tags
    keywords_in_tags = sum(1 for kw in keywords[:10] if any(kw in tag.lower() for tag in tags))
    tag_score += min(30.0, keywords_in_tags * 5.0)

    # Keyword density
    if keywords:
        primary_keyword = keywords[0]
        all_text = f"{title} {description}".lower()
        total_words = len(all_text.split())
        keyword_count = all_text.count(primary_keyword)
        keyword_density = keyword_count / total_words if total_words > 0 else 0.0

        if 0.01 <= keyword_density <= 0.05:
            tag_score += 10.0  # Good density
        elif keyword_density > 0.05:
            recommendations.append("Keyword density too high (keyword stuffing)")

    # Calculate overall score (0-100)
    raw_score = title_score + desc_score + tag_score
    overall = min(100.0, raw_score)

    if overall < 50:
        recommendations.append("SEO score is low - consider improving title and description")

    return SEOScore(
        title_score=min(100.0, title_score),
        description_score=min(100.0, desc_score),
        tag_score=min(100.0, tag_score),
        keyword_density=round(keyword_density, 4),
        overall_score=round(overall, 1),
        recommendations=recommendations,
    )


# ── Hashtag Generation ────────────────────────────────────────

def _generate_hashtags(
    keywords: list[str],
    category: str,
    mood: str,
    platform: str = "youtube",
    count: int = 10,
) -> list[str]:
    """Generate hashtags with relevance scoring.

    Combines keyword-based hashtags with category, mood, and
    platform-specific hashtags.

    Args:
        keywords: List of keyword strings.
        category: Content category.
        mood: Content mood.
        platform: Target platform.
        count: Number of hashtags to generate.

    Returns:
        List of hashtag strings (without # prefix).
    """
    hashtags: list[str] = []

    # Keyword-based hashtags
    for kw in keywords[:5]:
        hashtag = kw.replace(" ", "").replace("_", "")
        if len(hashtag) >= 3:
            hashtags.append(hashtag)

    # Category hashtags
    category_hashtags: dict[str, list[str]] = {
        "tech": ["TechTok", "CodingLife", "DevCommunity", "TechTips"],
        "gaming": ["GamingTikTok", "GamerLife", "GameReview", "GamingCommunity"],
        "cooking": ["FoodTok", "CookingTips", "RecipeOfTheDay", "Foodie"],
        "fitness": ["FitTok", "WorkoutMotivation", "FitnessTips", "GymLife"],
        "music": ["MusicTok", "NewMusic", "Musician", "SongCover"],
        "travel": ["TravelTok", "Wanderlust", "TravelTips", "Explore"],
        "comedy": ["ComedyTok", "FunnyVideos", "Humor", "LOL"],
        "education": ["EduTok", "LearnOnTikTok", "StudyTips", "Education"],
        "finance": ["FinTok", "MoneyTips", "Investing", "PersonalFinance"],
        "fashion": ["FashionTok", "OOTD", "StyleInspo", "BeautyTips"],
        "sports": ["SportsTok", "GameDay", "Athlete", "SportsHighlights"],
    }

    if category in category_hashtags:
        hashtags.extend(category_hashtags[category][:3])

    # Platform-specific hashtags
    platform_hashtags: dict[str, list[str]] = {
        "youtube": ["Shorts", "YouTubeShorts", "Youtuber"],
        "tiktok": ["fyp", "foryou", "viral", "trending"],
        "reels": ["Reels", "ReelsInstagram", "InstaReels"],
    }

    if platform in platform_hashtags:
        hashtags.extend(platform_hashtags[platform][:3])

    # Mood hashtags
    mood_hashtags: dict[str, list[str]] = {
        "energetic": ["Motivation", "Energy", "Hype"],
        "funny": ["Funny", "LOL", "Comedy"],
        "educational": ["LearnSomethingNew", "DidYouKnow", "Facts"],
        "inspirational": ["Inspiration", "Motivation", "Mindset"],
    }

    if mood in mood_hashtags:
        hashtags.extend(mood_hashtags[mood][:2])

    # Deduplicate and limit
    seen: set[str] = set()
    unique_hashtags: list[str] = []
    for tag in hashtags:
        tag_lower = tag.lower()
        if tag_lower not in seen:
            seen.add(tag_lower)
            unique_hashtags.append(tag)

    return unique_hashtags[:count]


# ── Description Generation ────────────────────────────────────

def _generate_description(
    hook: str,
    keywords: list[str],
    original_title: str,
    uploader: str,
    hashtags: list[str],
    category: str,
    mood: str,
    platform: str = "youtube",
) -> str:
    """Generate a platform-optimized description.

    Includes hook sentence, key points, hashtags, and CTA.

    Args:
        hook: Opening hook sentence.
        keywords: List of keyword strings.
        original_title: Original video title.
        uploader: Video uploader name.
        hashtags: List of hashtag strings.
        category: Content category.
        mood: Content mood.
        platform: Target platform.

    Returns:
        Formatted description string.
    """
    parts: list[str] = []

    # Hook sentence
    if hook:
        parts.append(hook)
        parts.append("")

    # Key points from keywords
    if keywords:
        parts.append("Key Topics:")
        for kw in keywords[:5]:
            parts.append(f"  \u2022 {kw.title()}")
        parts.append("")

    # Source attribution
    parts.append(f"Clipped from: {original_title}")
    if uploader:
        parts.append(f"by {uploader}")
    parts.append("")

    # CTA
    cta_options: dict[str, str] = {
        "youtube": "Subscribe for more shorts! \U0001f514",
        "tiktok": "Follow for more! \u2764\ufe0f",
        "reels": "Follow for more! \u2764\ufe0f",
    }
    parts.append(cta_options.get(platform, "Follow for more!"))
    parts.append("")

    # Hashtags
    if hashtags:
        hashtag_str = " ".join(f"#{tag}" for tag in hashtags)
        parts.append(hashtag_str)

    return "\n".join(parts)


# ── Trending Keyword Detection ────────────────────────────────

def _detect_trending(keywords: list[str]) -> list[str]:
    """Identify keywords that might be currently trending.

    Uses simple heuristics based on common trending patterns
    (seasonal, cultural, technological).

    Args:
        keywords: List of keyword strings.

    Returns:
        List of potentially trending keywords.
    """
    # Simulated trending indicators (in production, would use real trend data)
    trending_signals: set[str] = {
        "ai", "chatgpt", "gpt", "openai", "tiktok", "iphone",
        "trump", "biden", "election", "crypto", "bitcoin", "nft",
        "metaverse", "vr", "twitter", "x", "threads", "instagram",
        "youtube", "shorts", "viral", "trending", "fyp",
        "remote", "work", "layoffs", "inflation", "recession",
        "climate", "sustainability", "vegan", "plant", "based",
    }

    trending = []
    for kw in keywords:
        if kw.lower() in trending_signals:
            trending.append(kw)

    return trending


# ── Emoji Recommendation ──────────────────────────────────────

def _recommend_emojis(keywords: list[str], category: str) -> list[str]:
    """Recommend emojis based on content category and keywords.

    Args:
        keywords: List of keyword strings.
        category: Content category.

    Returns:
        List of emoji strings.
    """
    emojis: list[str] = []

    # Category-based emojis
    if category in _CATEGORY_EMOJIS:
        emojis.extend(_CATEGORY_EMOJIS[category][:2])

    # Keyword-based emojis
    for kw in keywords[:5]:
        if kw in _EMOJI_MAP:
            emoji_key = _EMOJI_MAP[kw]
            emoji = _EMOJI_UNICODE.get(emoji_key, "")
            if emoji and emoji not in emojis:
                emojis.append(emoji)

    # Default emoji if none found
    if not emojis:
        emojis.append("\U0001f3ac")  # Clapper board

    return emojis[:4]


# ── Main Metadata Generation Function ─────────────────────────

def generate_metadata(
    video_info: dict[str, Any],
    transcription: TranscriptionResult,
    settings: Settings | None = None,
) -> MetadataResult:
    """Generate platform-specific metadata from video info and transcription.

    Full pipeline: TF-IDF keyword extraction -> bigram/trigram extraction ->
    content categorization -> mood detection -> audience detection ->
    multi-strategy title generation -> description generation ->
    hashtag generation -> SEO scoring -> emoji recommendation.

    All generation is done locally using keyword extraction and templates —
    no external API calls are made.

    Args:
        video_info: Dict with video metadata (title, uploader, tags, etc.).
        transcription: TranscriptionResult with word timestamps.
        settings: Optional Settings override.

    Returns:
        MetadataResult with titles, descriptions, captions, and keywords.
    """
    if settings is None:
        settings = get_settings()

    original_title = video_info.get("title", "Untitled Video")
    uploader = video_info.get("uploader", "")
    original_tags = video_info.get("tags", [])
    language = getattr(settings, "METADATA_LANGUAGE", "en")

    # ── Step 1: Extract keywords with TF-IDF ─────────────
    title_words = original_title.split() if original_title else []
    keyword_scores = _extract_keywords_tfidf(
        transcription,
        video_tags=original_tags,
        title_words=title_words,
        top_n=settings.METADATA_MAX_KEYWORDS,
        language=language,
    )
    keywords = [ks.keyword for ks in keyword_scores]

    # ── Step 2: Extract bigrams and trigrams ──────────────
    bigrams = _extract_ngrams(transcription, n=2, top_n=10, language=language)
    trigrams = _extract_ngrams(transcription, n=3, top_n=5, language=language)

    # Add top bigrams to keywords
    for bigram, count in bigrams[:3]:
        if bigram not in keywords:
            keywords.append(bigram)

    # ── Step 3: Detect content category ──────────────────
    category, category_conf = _detect_category(keywords)
    logger.info("Content category: %s (%.0f%% confidence)", category, category_conf * 100)

    # ── Step 4: Detect mood/tone ─────────────────────────
    full_text = transcription.text
    mood, mood_conf = _detect_mood(full_text)
    logger.info("Mood detection: %s (%.0f%% confidence)", mood, mood_conf * 100)

    # ── Step 5: Detect target audience ───────────────────
    target_audience = _detect_audience(full_text)

    # ── Step 6: Detect trending keywords ─────────────────
    trending = _detect_trending(keywords)

    # ── Step 7: Generate emojis ──────────────────────────
    emoji_recs = _recommend_emojis(keywords, category)

    # ── Step 8: Generate YouTube title ───────────────────
    clean_title = _strip_title_suffixes(original_title)
    emoji = emoji_recs[0] if emoji_recs else "\U0001f3ac"

    # Primary title: hook + emoji + #Shorts
    hook_title = _generate_title_hook(transcription)
    if hook_title and len(hook_title) > 15:
        yt_title = f"{emoji} {hook_title}"
    else:
        if len(clean_title) > 60:
            yt_title = f"{emoji} {clean_title[:57]}..."
        else:
            yt_title = f"{emoji} {clean_title}"

    if len(yt_title) + 8 <= 100:
        yt_title += " #Shorts"
    yt_title = yt_title[:100]

    # ── Step 9: Generate A/B title variants ──────────────
    title_variants: list[str] = [yt_title]

    for strategy_name, strategy_fn in _TITLE_STRATEGIES.items():
        if strategy_name == "hook":
            continue  # Already used
        try:
            if strategy_name == "keyword":
                variant = strategy_fn(keyword_scores)
            else:
                variant = strategy_fn(keyword_scores)
            if variant and variant != yt_title:
                # Add emoji and #Shorts
                variant = f"{emoji} {variant}"
                if len(variant) + 8 <= 100:
                    variant += " #Shorts"
                variant = variant[:100]
                title_variants.append(variant)
        except Exception:
            pass

    # Keep only top 3 variants
    title_variants = title_variants[:3]

    # ── Step 10: Generate hashtags ────────────────────────
    hashtags = _generate_hashtags(
        keywords, category, mood, "youtube",
        count=settings.METADATA_HASHTAG_COUNT,
    )

    # ── Step 11: Generate hook ────────────────────────────
    hook = _generate_hook(transcription.segments, clean_title)

    # ── Step 12: Generate YouTube description ─────────────
    yt_description = _generate_description(
        hook, keywords, original_title, uploader,
        hashtags, category, mood, "youtube",
    )

    # ── Step 13: Generate TikTok caption ──────────────────
    first_sentence = transcription.segments[0].text if transcription.segments else clean_title
    tiktok_hashtags = _generate_hashtags(keywords, category, mood, "tiktok", count=5)
    tiktok_caption = f"{first_sentence} {' '.join(f'#{tag}' for tag in tiktok_hashtags)}"
    tiktok_caption = tiktok_caption[:2200]

    # ── Step 14: Generate Reels caption ───────────────────
    reels_hashtags = _generate_hashtags(keywords, category, mood, "reels", count=7)
    reels_caption = f"{first_sentence} {' '.join(f'#{tag}' for tag in reels_hashtags)}"
    reels_caption = reels_caption[:2200]

    # ── Step 15: Score SEO ────────────────────────────────
    seo = _score_seo(yt_title, yt_description, keywords[:20], keywords)

    # ── Step 16: Build result ─────────────────────────────
    result = MetadataResult(
        youtube_title=yt_title,
        youtube_description=yt_description,
        youtube_tags=keywords[:20],
        tiktok_caption=tiktok_caption,
        reels_caption=reels_caption,
        keywords=keywords,
        thumbnail_keywords=keywords[:5],
        title_variants=title_variants,
        seo_score=seo.overall_score,
        content_category=category,
        mood=mood,
        target_audience=target_audience,
        hashtags=hashtags,
        emoji_recommendations=emoji_recs,
    )

    # ── Step 17: Save metadata JSON ───────────────────────
    try:
        metadata_dir = settings.METADATA_DIR
        metadata_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = sanitize_filename(original_title, max_length=60)
        metadata_path = metadata_dir / f"{safe_name}_{timestamp}.json"

        metadata_json = {
            "youtube_title": result.youtube_title,
            "youtube_description": result.youtube_description,
            "youtube_tags": result.youtube_tags,
            "tiktok_caption": result.tiktok_caption,
            "reels_caption": result.reels_caption,
            "keywords": result.keywords,
            "title_variants": result.title_variants,
            "seo_score": result.seo_score,
            "content_category": result.content_category,
            "mood": result.mood,
            "target_audience": result.target_audience,
            "hashtags": result.hashtags,
            "emoji_recommendations": result.emoji_recommendations,
            "bigrams": [f"{bg} ({cnt})" for bg, cnt in bigrams],
            "trigrams": [f"{tg} ({cnt})" for tg, cnt in trigrams],
            "trending_keywords": trending,
            "original_title": original_title,
            "uploader": uploader,
            "seo_recommendations": seo.recommendations,
        }

        metadata_path.write_text(
            json.dumps(metadata_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Metadata saved: %s", metadata_path.name)
    except Exception as exc:
        logger.warning("Failed to save metadata JSON: %s", exc)

    logger.info(
        "Metadata generated: %d keywords, category=%s, mood=%s, SEO=%.0f, yt_title=%d chars",
        len(keywords), category, mood, seo.overall_score, len(yt_title),
    )
    return result


# ── Helper Functions ──────────────────────────────────────────

def _strip_title_suffixes(title: str) -> str:
    """Remove common YouTube title suffixes.

    Args:
        title: Original title string.

    Returns:
        Cleaned title string.
    """
    patterns = [
        r"\s*\|\s*Full\s+Video.*$",
        r"\s*\|\s*Full\s+Episode.*$",
        r"\s*Episode\s+\d+.*$",
        r"\s*Part\s+\d+.*$",
        r"\s*\(Full\).*$",
        r"\s*-\s*Full\s+Video.*$",
        r"\s*\[.*?\]\s*$",
        r"\s*\(Official\s+.*?\)\s*$",
        r"\s*#\w+\s*$",
    ]
    cleaned = title
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _generate_hook(segments: list, fallback: str) -> str:
    """Generate a compelling hook from the first transcription segments.

    Args:
        segments: List of Segment objects.
        fallback: Fallback text if no good hook found.

    Returns:
        Hook string.
    """
    hook_lines = []
    for seg in segments[:3]:
        text = seg.text.strip() if hasattr(seg, "text") else str(seg).strip()
        if text and len(text) > 10:
            hook_lines.append(text)
        if len(hook_lines) >= 2:
            break

    if hook_lines:
        hook = " ".join(hook_lines)
        if len(hook) > 150:
            hook = hook[:147] + "..."
        return hook
    return fallback
