"""
core/thumbnail_generator.py — Thumbnail generation with multiple styles, face-aware
composition, text rendering with effects, and platform optimization.

Uses Pillow (PIL) for all image operations. Supports 4 thumbnail styles:
modern (gradient + bold text), minimal (clean + subtle), bold (high contrast + big text),
cinematic (dark overlay + dramatic text). Supports 1280x720 (YouTube), 1080x1920 (Shorts),
1080x1080 (Instagram).
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.settings import Settings, get_settings
from utils.ffmpeg_utils import probe_video, run_ffmpeg, get_video_thumbnail, FFmpegError
from utils.file_utils import safe_delete, make_output_path
from utils.logger import get_logger

logger = get_logger("thumbnail_generator")


# ── Platform Size Presets ─────────────────────────────────────

PLATFORM_SIZES: dict[str, tuple[int, int]] = {
    "youtube": (1280, 720),      # YouTube standard thumbnail
    "youtube_shorts": (1080, 1920),  # YouTube Shorts vertical
    "instagram": (1080, 1080),   # Instagram square
    "instagram_reels": (1080, 1920),  # Instagram Reels vertical
    "tiktok": (1080, 1920),      # TikTok vertical
    "twitter": (1280, 720),      # Twitter card
    "facebook": (1280, 720),     # Facebook share
}

# Font fallbacks for different systems
_FONT_FALLBACKS: list[str] = [
    "Arial Black", "Arial", "Helvetica", "DejaVu Sans Bold",
    "Liberation Sans Bold", "Impact", "Roboto", "Noto Sans Bold",
]


def _get_font(size: int, bold: bool = True):
    """Get a PIL ImageFont with fallback for different systems.

    Args:
        size: Font size in pixels.
        bold: Whether to use bold variant.

    Returns:
        PIL ImageFont object.
    """
    from PIL import ImageFont

    for font_name in _FONT_FALLBACKS:
        try:
            font = ImageFont.truetype(font_name, size)
            return font
        except (OSError, IOError):
            continue

    # Final fallback to default font
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


# ── Data Classes ──────────────────────────────────────────────

@dataclass
class ThumbnailScore:
    """Score for thumbnail quality assessment."""

    brightness: float = 0.0
    contrast: float = 0.0
    face_presence: float = 0.0
    text_readability: float = 0.0
    overall: float = 0.0
    recommendations: list[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


# ── Frame Extraction ──────────────────────────────────────────

class ThumbnailGenerator:
    """Thumbnail generator with multiple styles and platform optimization.

    Provides methods for extracting best frames, generating styled thumbnails,
    adding text overlays with effects, and scoring thumbnail quality.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def extract_best_frames(
        self,
        video_path: Path,
        count: int = 3,
        strategy: str = "peak_energy",
    ) -> list[Path]:
        """Extract the best frames from a video for thumbnail candidates.

        Strategies:
        - 'peak_energy': Frames at audio energy peaks.
        - 'uniform': Evenly spaced frames.
        - 'scene_change': Frames at scene change points.

        Args:
            video_path: Path to the video file.
            count: Number of frames to extract.
            strategy: Extraction strategy.

        Returns:
            List of paths to extracted JPEG frames.

        Raises:
            FileNotFoundError: If video doesn't exist.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_info = probe_video(video_path)
        duration = video_info.duration
        extracted: list[Path] = []

        if strategy == "uniform":
            timestamps = [
                duration * (i + 1) / (count + 1)
                for i in range(count)
            ]
        elif strategy == "scene_change":
            try:
                from utils.ffmpeg_utils import detect_scene_changes
                scene_times = detect_scene_changes(video_path)
                if scene_times:
                    # Pick evenly from scene changes
                    step = max(1, len(scene_times) // count)
                    timestamps = scene_times[::step][:count]
                    # Pad if not enough scene changes
                    while len(timestamps) < count:
                        timestamps.append(duration * (len(timestamps) + 1) / (count + 1))
                else:
                    timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]
            except Exception:
                timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]
        else:  # peak_energy
            # Extract frames at energy peaks using audio analysis
            try:
                from utils.ffmpeg_utils import extract_audio_samples
                samples = extract_audio_samples(video_path, 2.0)
                if samples:
                    # Find peak energy timestamps
                    indexed = list(enumerate(samples))
                    indexed.sort(key=lambda x: x[1], reverse=True)
                    # Pick top N, ensuring they're spread out
                    selected_indices: list[int] = []
                    for idx, _ in indexed:
                        # Ensure at least 5 seconds between selected frames
                        if all(abs(idx - s) > 2 for s in selected_indices):
                            selected_indices.append(idx)
                        if len(selected_indices) >= count:
                            break
                    timestamps = [
                        idx * 2.0 for idx in sorted(selected_indices[:count])
                    ]
                    while len(timestamps) < count:
                        timestamps.append(duration * (len(timestamps) + 1) / (count + 1))
                else:
                    timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]
            except Exception:
                timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]

        for i, ts in enumerate(timestamps):
            ts = max(0.5, min(ts, duration - 0.5))
            tmp_path = Path(tempfile.mktemp(suffix=f"_thumb_{i}.jpg"))
            try:
                get_video_thumbnail(video_path, ts, tmp_path)
                if tmp_path.exists():
                    extracted.append(tmp_path)
            except Exception as exc:
                logger.debug("Failed to extract frame at %.1fs: %s", ts, exc)

        logger.info("Extracted %d thumbnail frames from %s", len(extracted), video_path.name)
        return extracted

    def generate_thumbnail(
        self,
        frame_path: Path,
        title_text: str,
        output_path: Path,
        style: str = "modern",
        platform: str = "youtube",
        logo_path: Path | None = None,
    ) -> Path:
        """Generate a styled thumbnail from a frame image.

        Styles:
        - 'modern': Gradient overlay + bold centered text.
        - 'minimal': Clean image + subtle bottom text.
        - 'bold': High contrast + big text with outline.
        - 'cinematic': Dark vignette + dramatic text.

        Args:
            frame_path: Path to the source frame image.
            title_text: Title text to overlay.
            output_path: Destination path.
            style: Thumbnail style name.
            platform: Target platform for sizing.
            logo_path: Optional channel logo.

        Returns:
            Path to the generated thumbnail image.
        """
        from PIL import Image, ImageDraw, ImageFilter, ImageFont

        if not frame_path.exists():
            logger.error("Frame not found: %s", frame_path)
            return output_path

        # Get target size
        target_w, target_h = PLATFORM_SIZES.get(platform, (1280, 720))

        # Load and resize frame
        img = Image.open(frame_path).convert("RGB")
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        if style == "modern":
            img = self._apply_modern_style(img, title_text, target_w, target_h)
        elif style == "minimal":
            img = self._apply_minimal_style(img, title_text, target_w, target_h)
        elif style == "bold":
            img = self._apply_bold_style(img, title_text, target_w, target_h)
        elif style == "cinematic":
            img = self._apply_cinematic_style(img, title_text, target_w, target_h)
        else:
            img = self._apply_modern_style(img, title_text, target_w, target_h)

        # Add logo if provided
        if logo_path and logo_path.exists():
            img = self._add_logo_to_thumbnail(img, logo_path, target_w, target_h)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path), "JPEG", quality=95)
        logger.info("Thumbnail generated: %s (style=%s, %dx%d)",
                    output_path.name, style, target_w, target_h)
        return output_path

    def _apply_modern_style(
        self, img, title_text: str, w: int, h: int,
    ):
        """Apply modern style: gradient overlay + bold text.

        Args:
            img: PIL Image.
            title_text: Title string.
            w: Width.
            h: Height.

        Returns:
            Styled PIL Image.
        """
        from PIL import Image, ImageDraw, ImageFilter, ImageFont

        # Create gradient overlay (dark at bottom, transparent at top)
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Bottom gradient (60% of height)
        gradient_start = int(h * 0.4)
        for y in range(gradient_start, h):
            alpha = int(220 * (y - gradient_start) / (h - gradient_start))
            draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))

        # Convert base image to RGBA for compositing
        img_rgba = img.convert("RGBA")
        img_rgba = Image.alpha_composite(img_rgba, overlay)

        # Add text
        draw = ImageDraw.Draw(img_rgba)
        font_size = min(w // 12, 72)
        font = _get_font(font_size, bold=True)

        # Word wrap the title
        lines = self._word_wrap(title_text, font, w - 80)

        y_offset = h - len(lines) * (font_size + 10) - 60
        for line in lines:
            # Text shadow
            draw.text((42, y_offset + 3), line, fill=(0, 0, 0, 200), font=font)
            # Main text
            draw.text((40, y_offset), line, fill=(255, 255, 255, 255), font=font)
            y_offset += font_size + 10

        return img_rgba.convert("RGB")

    def _apply_minimal_style(
        self, img, title_text: str, w: int, h: int,
    ):
        """Apply minimal style: clean image + subtle bottom text bar.

        Args:
            img: PIL Image.
            title_text: Title string.
            w: Width.
            h: Height.

        Returns:
            Styled PIL Image.
        """
        from PIL import Image, ImageDraw, ImageFont

        # Subtle bottom bar
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        bar_height = 120
        for y in range(h - bar_height, h):
            alpha = int(180 * (y - (h - bar_height)) / bar_height)
            draw.line([(0, y), (w, y)], fill=(255, 255, 255, alpha))

        img_rgba = img.convert("RGBA")
        img_rgba = Image.alpha_composite(img_rgba, overlay)

        # Add text
        draw = ImageDraw.Draw(img_rgba)
        font_size = min(w // 16, 48)
        font = _get_font(font_size, bold=False)

        lines = self._word_wrap(title_text, font, w - 60)
        y_offset = h - len(lines) * (font_size + 8) - 30
        for line in lines:
            draw.text((30, y_offset), line, fill=(20, 20, 20, 230), font=font)
            y_offset += font_size + 8

        return img_rgba.convert("RGB")

    def _apply_bold_style(
        self, img, title_text: str, w: int, h: int,
    ):
        """Apply bold style: high contrast + big text with thick outline.

        Args:
            img: PIL Image.
            title_text: Title string.
            w: Width.
            h: Height.

        Returns:
            Styled PIL Image.
        """
        from PIL import Image, ImageDraw, ImageEnhance, ImageFont

        # Increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.4)
        # Increase saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.3)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Center dark overlay
        center_h = int(h * 0.4)
        for y in range(int(h * 0.25), int(h * 0.25) + center_h):
            alpha = int(160)
            draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))

        img_rgba = img.convert("RGBA")
        img_rgba = Image.alpha_composite(img_rgba, overlay)

        # Big bold text with outline
        draw = ImageDraw.Draw(img_rgba)
        font_size = min(w // 8, 96)
        font = _get_font(font_size, bold=True)

        lines = self._word_wrap(title_text, font, w - 100)
        total_text_height = len(lines) * (font_size + 12)
        y_offset = (h - total_text_height) // 2

        # Draw outline (stroke)
        outline_width = 4
        for line in lines:
            for adj_x in range(-outline_width, outline_width + 1):
                for adj_y in range(-outline_width, outline_width + 1):
                    draw.text(
                        (50 + adj_x, y_offset + adj_y),
                        line, fill=(0, 0, 0, 255), font=font,
                    )
            # Main text in yellow
            draw.text((50, y_offset), line, fill=(255, 255, 0, 255), font=font)
            y_offset += font_size + 12

        return img_rgba.convert("RGB")

    def _apply_cinematic_style(
        self, img, title_text: str, w: int, h: int,
    ):
        """Apply cinematic style: dark vignette + dramatic text.

        Args:
            img: PIL Image.
            title_text: Title string.
            w: Width.
            h: Height.

        Returns:
            Styled PIL Image.
        """
        from PIL import Image, ImageDraw, ImageFilter, ImageFont

        # Vignette effect
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Radial vignette
        center_x, center_y = w // 2, h // 2
        max_dist = math.sqrt(center_x ** 2 + center_y ** 2)

        for y in range(0, h, 2):  # Step by 2 for performance
            for x in range(0, w, 2):
                dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                alpha = int(min(255, 200 * (dist / max_dist) ** 1.5))
                draw.point((x, y), fill=(0, 0, 0, alpha))
                draw.point((x + 1, y), fill=(0, 0, 0, alpha))
                draw.point((x, y + 1), fill=(0, 0, 0, alpha))
                draw.point((x + 1, y + 1), fill=(0, 0, 0, alpha))

        # Bottom gradient for text
        for y in range(int(h * 0.6), h):
            alpha = int(220 * (y - h * 0.6) / (h * 0.4))
            draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))

        img_rgba = img.convert("RGBA")
        img_rgba = Image.alpha_composite(img_rgba, overlay)

        # Dramatic text
        draw = ImageDraw.Draw(img_rgba)
        font_size = min(w // 10, 80)
        font = _get_font(font_size, bold=True)

        lines = self._word_wrap(title_text, font, w - 80)
        y_offset = h - len(lines) * (font_size + 12) - 80

        for line in lines:
            # Shadow
            draw.text((42, y_offset + 3), line, fill=(0, 0, 0, 200), font=font)
            # White text with subtle glow
            draw.text((40, y_offset), line, fill=(255, 255, 255, 255), font=font)
            y_offset += font_size + 12

        # Add subtle letterbox bars
        bar_h = 40
        for y in range(bar_h):
            draw.line([(0, y), (w, y)], fill=(0, 0, 0, 255))
            draw.line([(0, h - y - 1), (w, h - y - 1)], fill=(0, 0, 0, 255))

        return img_rgba.convert("RGB")

    def add_text_overlay(
        self,
        image_path: Path,
        text: str,
        position: str = "bottom",
        font_size: int = 48,
        color: str = "white",
        shadow: bool = True,
        outline: bool = True,
    ) -> Path:
        """Add text overlay to an image with shadow, outline, and gradient effects.

        Args:
            image_path: Path to the source image.
            text: Text to overlay.
            position: Position ('bottom', 'top', 'center').
            font_size: Font size in pixels.
            color: Text color name or hex.
            shadow: Add text shadow.
            outline: Add text outline.

        Returns:
            Path to the modified image (overwrites original).
        """
        from PIL import Image, ImageDraw, ImageFont

        if not image_path.exists():
            logger.error("Image not found: %s", image_path)
            return image_path

        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        font = _get_font(font_size, bold=True)

        # Parse color
        text_color = self._parse_color(color)

        # Word wrap
        lines = self._word_wrap(text, font, w - 60)
        line_height = font_size + 8
        total_height = len(lines) * line_height

        # Calculate Y position
        if position == "top":
            y_offset = 40
        elif position == "center":
            y_offset = (h - total_height) // 2
        else:  # bottom
            y_offset = h - total_height - 40

        for line in lines:
            # Calculate X for centering
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            x = (w - text_w) // 2

            # Outline
            if outline:
                for adj_x in range(-3, 4):
                    for adj_y in range(-3, 4):
                        draw.text(
                            (x + adj_x, y_offset + adj_y),
                            line, fill=(0, 0, 0, 255), font=font,
                        )

            # Shadow
            if shadow:
                draw.text((x + 2, y_offset + 2), line, fill=(0, 0, 0, 180), font=font)

            # Main text
            draw.text((x, y_offset), line, fill=text_color, font=font)
            y_offset += line_height

        img.convert("RGB").save(str(image_path), "JPEG", quality=95)
        return image_path

    def add_channel_branding(
        self,
        image_path: Path,
        logo_path: Path,
        output_path: Path,
    ) -> Path:
        """Add channel branding (logo) to a thumbnail image.

        Places the logo in the top-right corner with a slight margin.

        Args:
            image_path: Path to the thumbnail image.
            logo_path: Path to the channel logo.
            output_path: Destination path.

        Returns:
            Path to the branded thumbnail.
        """
        from PIL import Image

        if not image_path.exists() or not logo_path.exists():
            logger.error("Image or logo not found")
            return output_path

        img = Image.open(image_path).convert("RGBA")
        logo = Image.open(logo_path).convert("RGBA")

        w, h = img.size
        # Logo size: ~15% of thumbnail width
        logo_size = int(w * 0.15)
        logo_aspect = logo.height / logo.width if logo.width > 0 else 1.0
        logo_h = int(logo_size * logo_aspect)
        logo = logo.resize((logo_size, logo_h), Image.Resampling.LANCZOS)

        # Position: top-right with margin
        margin = 20
        logo_x = w - logo_size - margin
        logo_y = margin

        img.paste(logo, (logo_x, logo_y), logo)
        img.convert("RGB").save(str(output_path), "JPEG", quality=95)

        logger.info("Channel branding added: %s", output_path.name)
        return output_path

    def compose_thumbnail(
        self,
        background_path: Path,
        face_crop_path: Path,
        title: str,
        logo_path: Path | None = None,
        output_path: Path = None,
    ) -> Path:
        """Compose a thumbnail with background, face crop, title, and logo.

        Creates a face-aware composition: face crop on one side,
        background on the other, with title text.

        Args:
            background_path: Path to the background image.
            face_crop_path: Path to a face-closeup image.
            title: Title text to overlay.
            logo_path: Optional channel logo path.
            output_path: Destination path (auto-generated if None).

        Returns:
            Path to the composed thumbnail.
        """
        from PIL import Image, ImageDraw, ImageFont

        if not background_path.exists():
            logger.error("Background image not found: %s", background_path)
            return output_path or background_path

        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix="_composed.jpg"))

        w, h = 1280, 720
        bg = Image.open(background_path).convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)

        if face_crop_path.exists():
            face = Image.open(face_crop_path).convert("RGBA")
            # Face on the right side
            face_w = int(w * 0.45)
            face_aspect = face.height / face.width if face.width > 0 else 1.0
            face_h = int(face_w * face_aspect)
            face = face.resize((face_w, min(face_h, h)), Image.Resampling.LANCZOS)

            # Paste face on the right
            face_x = w - face_w
            face_y = (h - face.height) // 2
            bg.paste(face, (face_x, face_y), face)

        # Gradient overlay on left side for text
        draw = ImageDraw.Draw(bg)
        for x in range(int(w * 0.55)):
            alpha = int(200 * (1 - x / (w * 0.55)))
            draw.line([(x, 0), (x, h)], fill=(0, 0, 0, alpha))

        # Title text
        font_size = min(w // 14, 60)
        font = _get_font(font_size, bold=True)
        lines = self._word_wrap(title, font, int(w * 0.5))

        y_offset = h // 3
        for line in lines:
            draw.text((42, y_offset + 2), line, fill=(0, 0, 0, 200), font=font)
            draw.text((40, y_offset), line, fill=(255, 255, 255, 255), font=font)
            y_offset += font_size + 8

        # Add logo
        if logo_path and logo_path.exists():
            logo = Image.open(logo_path).convert("RGBA")
            logo_size = int(w * 0.1)
            logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
            bg.paste(logo, (40, h - logo_size - 30), logo)

        bg.convert("RGB").save(str(output_path), "JPEG", quality=95)
        logger.info("Thumbnail composed: %s", output_path.name)
        return output_path

    def create_comparison_thumbnail(
        self,
        before_path: Path,
        after_path: Path,
        output_path: Path,
        label_before: str = "BEFORE",
        label_after: str = "AFTER",
    ) -> Path:
        """Create a before/after comparison thumbnail.

        Splits the thumbnail into two halves with labels.

        Args:
            before_path: Path to the "before" image.
            after_path: Path to the "after" image.
            output_path: Destination path.
            label_before: Label for the before side.
            label_after: Label for the after side.

        Returns:
            Path to the comparison thumbnail.
        """
        from PIL import Image, ImageDraw, ImageFont

        w, h = 1280, 720
        half_w = w // 2

        result = Image.new("RGB", (w, h))

        if before_path.exists():
            before = Image.open(before_path).convert("RGB").resize((half_w, h), Image.Resampling.LANCZOS)
            result.paste(before, (0, 0))

        if after_path.exists():
            after = Image.open(after_path).convert("RGB").resize((half_w, h), Image.Resampling.LANCZOS)
            result.paste(after, (half_w, 0))

        # Draw divider line
        draw = ImageDraw.Draw(result)
        draw.line([(half_w, 0), (half_w, h)], fill=(255, 255, 0), width=4)

        # Labels
        font = _get_font(36, bold=True)
        # Before label
        draw.text((20, 20), label_before, fill=(255, 255, 255), font=font)
        draw.text((22, 22), label_before, fill=(0, 0, 0), font=font)
        # After label
        draw.text((half_w + 20, 20), label_after, fill=(255, 255, 255), font=font)
        draw.text((half_w + 22, 22), label_after, fill=(0, 0, 0), font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(output_path), "JPEG", quality=95)
        logger.info("Comparison thumbnail created: %s", output_path.name)
        return output_path

    def score_thumbnail(self, image_path: Path) -> ThumbnailScore:
        """Score a thumbnail's quality based on multiple signals.

        Evaluates brightness, contrast, face presence, and text readability.

        Args:
            image_path: Path to the thumbnail image.

        Returns:
            ThumbnailScore with breakdown and recommendations.
        """
        from PIL import Image
        import numpy as np

        if not image_path.exists():
            return ThumbnailScore(recommendations=["Image file not found"])

        try:
            img = Image.open(image_path).convert("RGB")
            arr = np.array(img, dtype=float)
        except Exception as exc:
            return ThumbnailScore(recommendations=[f"Cannot open image: {exc}"])

        h, w, _ = arr.shape
        recs: list[str] = []

        # Brightness score (optimal: 100-180 on 0-255 scale)
        mean_brightness = np.mean(arr)
        if 100 <= mean_brightness <= 180:
            brightness_score = 80.0
        elif 80 <= mean_brightness <= 200:
            brightness_score = 50.0
        else:
            brightness_score = 20.0
            if mean_brightness < 80:
                recs.append("Thumbnail is too dark - increase brightness")
            elif mean_brightness > 200:
                recs.append("Thumbnail is too bright - reduce brightness")

        # Contrast score (standard deviation of luminance)
        luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        contrast = np.std(luminance)
        if contrast > 60:
            contrast_score = 80.0
        elif contrast > 40:
            contrast_score = 60.0
        else:
            contrast_score = 30.0
            recs.append("Low contrast - consider increasing contrast for visibility")

        # Face presence (try OpenCV)
        face_score = 0.0
        try:
            import cv2
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                face_score = 80.0
                # Check face size relative to image
                fx, fy, fw, fh = faces[0]
                face_area_ratio = (fw * fh) / (w * h)
                if face_area_ratio > 0.05:
                    face_score = 90.0
                elif face_area_ratio > 0.02:
                    face_score = 70.0
            else:
                face_score = 30.0
                recs.append("No face detected - faces in thumbnails increase CTR")
        except ImportError:
            face_score = 50.0  # Neutral if OpenCV unavailable

        # Text readability (estimate based on brightness variation in bottom third)
        bottom_third = arr[int(h * 0.7):, :, :]
        bottom_std = np.std(bottom_third)
        text_score = min(80.0, bottom_std * 2) if bottom_std < 40 else 60.0
        if bottom_std < 30:
            recs.append("Bottom area lacks contrast for text readability")

        # Overall score (weighted average)
        overall = (
            brightness_score * 0.2 +
            contrast_score * 0.25 +
            face_score * 0.3 +
            text_score * 0.25
        )

        return ThumbnailScore(
            brightness=round(brightness_score, 1),
            contrast=round(contrast_score, 1),
            face_presence=round(face_score, 1),
            text_readability=round(text_score, 1),
            overall=round(overall, 1),
            recommendations=recs,
        )

    def optimize_for_platform(
        self,
        image_path: Path,
        platform: str = "youtube",
    ) -> Path:
        """Resize and optimize a thumbnail for a specific platform.

        Args:
            image_path: Path to the source image.
            platform: Target platform.

        Returns:
            Path to the optimized image (overwrites original).
        """
        from PIL import Image

        if not image_path.exists():
            logger.error("Image not found: %s", image_path)
            return image_path

        target_w, target_h = PLATFORM_SIZES.get(platform, (1280, 720))

        img = Image.open(image_path).convert("RGB")
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img.save(str(image_path), "JPEG", quality=95)

        logger.info("Thumbnail optimized for %s: %dx%d", platform, target_w, target_h)
        return image_path

    # ── Private Helper Methods ───────────────────────────

    def _word_wrap(
        self,
        text: str,
        font,
        max_width: int,
    ) -> list[str]:
        """Word-wrap text to fit within a given width.

        Args:
            text: Text to wrap.
            font: PIL ImageFont to measure with.
            max_width: Maximum line width in pixels.

        Returns:
            List of text lines.
        """
        from PIL import ImageDraw

        words = text.split()
        if not words:
            return [""]

        lines: list[str] = []
        current_line = words[0]

        # Create a dummy image for text measurement
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)

        for word in words[1:]:
            test_line = current_line + " " + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)

        # Limit to 3 lines
        if len(lines) > 3:
            lines = lines[:3]
            lines[-1] = lines[-1][: -3] + "..."

        return lines

    @staticmethod
    def _parse_color(color: str) -> tuple[int, int, int, int]:
        """Parse a color string into RGBA tuple.

        Args:
            color: Color name or hex string.

        Returns:
            RGBA tuple.
        """
        color_map: dict[str, tuple[int, int, int, int]] = {
            "white": (255, 255, 255, 255),
            "black": (0, 0, 0, 255),
            "red": (255, 0, 0, 255),
            "yellow": (255, 255, 0, 255),
            "green": (0, 255, 0, 255),
            "blue": (0, 0, 255, 255),
            "cyan": (0, 255, 255, 255),
            "magenta": (255, 0, 255, 255),
        }

        if color.lower() in color_map:
            return color_map[color.lower()]

        # Try hex parsing
        try:
            hex_str = color.lstrip("#")
            if len(hex_str) == 6:
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                return (r, g, b, 255)
        except (ValueError, IndexError):
            pass

        return (255, 255, 255, 255)  # Default white

    def _add_logo_to_thumbnail(
        self,
        img,
        logo_path: Path,
        w: int,
        h: int,
    ):
        """Add a channel logo to the thumbnail.

        Args:
            img: PIL Image (RGBA).
            logo_path: Path to the logo file.
            w: Image width.
            h: Image height.

        Returns:
            PIL Image with logo.
        """
        from PIL import Image

        logo = Image.open(logo_path).convert("RGBA")
        logo_size = int(w * 0.12)
        logo_aspect = logo.height / logo.width if logo.width > 0 else 1.0
        logo_h = int(logo_size * logo_aspect)
        logo = logo.resize((logo_size, logo_h), Image.Resampling.LANCZOS)

        # Top-right corner
        margin = 15
        x = w - logo_size - margin
        y = margin

        img.paste(logo, (x, y), logo)
        return img
