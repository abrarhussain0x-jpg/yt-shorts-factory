# YT Shorts Factory

**Enterprise-grade YouTube Shorts automation pipeline — fully local, zero cloud cost, zero API fees.**

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![FFmpeg Required](https://img.shields.io/badge/FFmpeg-%3E%3D4.0-orange)

**Developed by Abrar Hussain**

YT Shorts Factory is a production-ready CLI system that downloads YouTube videos, extracts the highest-engagement segments using multi-signal analysis (audio energy, scene change, face tracking, motion detection), converts them to platform-optimised 9:16 Shorts, burns animated subtitles with word-level karaoke highlighting via Whisper AI, stamps a channel logo with cinematic fade, applies channel branding patterns, and outputs upload-ready files for YouTube Shorts, TikTok, and Instagram Reels — all locally on your machine.

## Features

- **Interactive TUI Menu** — Arrow-key navigable menus, no need to memorize CLI flags
- **Superfast Mode** — Single-pass FFmpeg pipeline, 4-8x faster than standard
- **Turbo Mode** — Ultrafast encoding, tiny Whisper, skip extras (3-4x faster)
- **Smart Download** — yt-dlp with format selection, duplicate detection, and retry logic
- **Multi-Signal Analysis** — Audio RMS energy + scene change + face tracking + motion detection
- **Smart Crop & Reframe** — Face-tracking-aware 9:16 crop with Lanczos scaling
- **Whisper AI Transcription** — Word-level timestamps, faster-whisper support (4x faster)
- **12 Subtitle Animations** — Karaoke, fade, pop, glow, typewriter, bounce, wave, rainbow, neon, matrix, 3d_rotate, none
- **Logo Overlay** — PNG stamp with configurable position, opacity, scale, and cinematic fade-in
- **Channel Branding Patterns** — 10+ built-in patterns (viral_hype, chill_vibes, news_alert, etc.)
- **Hook Generator** — Attention-grabbing first 3 seconds (text flash, zoom pulse, bold statement)
- **CTA Overlays** — Subscribe/Like prompts at optimal engagement moments
- **Audio Enhancement** — Noise reduction, dynamic compression, loudness normalization (EBU R128)
- **Multi-Platform Export** — YouTube Shorts, TikTok, Instagram Reels (parallel export)
- **Duration Presets** — 25s (quick), 45s (standard), 180s (3 min extended)
- **Multiple Aspect Ratios** — 9:16, 1:1, 4:5
- **Auto Metadata** — Rule-based title, description, and hashtag generation
- **Job Queue** — SQLite-backed persistent queue with retry and exponential backoff
- **Background Worker** — Multi-threaded daemon with auto-scaling based on system load
- **Analytics DB** — Full job history and video records with aggregate stats
- **Speed Optimizer** — Auto-detect GPU encoding (NVENC/AMF/QSV/VideoToolbox), optimal threads
- **Checkpoint Resume** — Resume crashed pipelines from the last completed step
- **GPU Acceleration** — CUDA/MPS for Whisper, NVENC/AMF for FFmpeg encoding
- **Rich CLI** — Colour-coded progress, step tracking, and summary panels

## Architecture

```
URL → Download → Analyze → Crop+Reframe → Transcribe → Subtitles → Logo → Export → Output
 │       │          │           │              │            │         │       │
 │       │          │           │              │            │         │       ├─ YouTube Shorts
 │       │          │           │              │            │         │       ├─ TikTok
 │       │          │           │              │            │         │       └─ Instagram Reels
 │       │          │           │              │            │         └─ Logo with fade
 │       │          │           │              │            └─ ASS animated subtitles
 │       │          │           │              └─ Whisper word timestamps
 │       │          │           └─ 9:16 smart crop + loudnorm
 │       │          └─ Multi-signal engagement scoring
 │       └─ yt-dlp with metadata
 └─ URL validation
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-repo/yt-shorts-factory.git
cd yt-shorts-factory
bash scripts/install_deps.sh

# 2. Configure
cp .env.example .env
# Edit .env with your settings (GPU, Whisper model, logo, etc.)

# 3. Run — Interactive Menu (recommended)
python main.py

# 4. Or use CLI directly
python main.py run --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Usage

### Interactive Mode (recommended)

Just run `python main.py` to launch the interactive TUI with arrow-key menus:

```bash
python main.py
```

This gives you a beautiful step-by-step guided workflow:
1. Choose action (Create Shorts, Quick Shorts, Batch, etc.)
2. Paste URL
3. Select duration, speed, platforms, animation, pattern
4. Confirm and run

### CLI Commands

```bash
# Process a single video
python main.py run --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Superfast mode (4-8x faster, single-pass FFmpeg)
python main.py run --url "URL" --superfast

# Turbo mode (3-4x faster, ultrafast encoding)
python main.py run --url "URL" --turbo

# Custom duration preset
python main.py run --url "URL" --preset 25s
python main.py run --url "URL" --preset 45s
python main.py run --url "URL" --preset 3min

# Channel branding pattern
python main.py run --url "URL" --pattern viral_hype

# Multiple platforms
python main.py run --url "URL" -p youtube -p tiktok -p reels

# Batch processing
python main.py run --batch urls.txt

# Job queue + worker
python main.py queue "URL"
python main.py worker --start

# History and stats
python main.py history --limit 10
python main.py stats --daily --channels

# Verify dependencies
python main.py verify

# Show duration presets
python main.py presets

# Show channel patterns
python main.py patterns
```

## Duration Presets

| Preset | Duration | Best For |
|--------|----------|----------|
| `quick` / `25s` | 25 seconds | TikTok fast scroll, attention grabbers |
| `standard` / `45s` | 45 seconds | YouTube Shorts & Reels sweet spot |
| `extended` / `3min` | 3 minutes | Long-form Shorts, full storytelling |
| Custom | Any seconds | `-d 30` for 30s, `-d 120` for 2 min |

## Channel Branding Patterns

| Pattern | Style | Best For |
|---------|-------|----------|
| `viral_hype` | Explosive hooks, neon text | Viral content, engagement |
| `chill_vibes` | Smooth fades, warm colors | Relaxing, ambient content |
| `news_alert` | Breaking news, ticker | News, updates |
| `educational` | Clean, structured | Tutorials, explainers |
| `gaming_clips` | Fast, glitch effects | Gaming highlights |
| `motivational` | Epic reveals, gold accents | Inspiration, quotes |
| `comedy_clip` | Punch timing, fun vibes | Funny moments |
| `lifestyle` | Aesthetic, minimal | Vlogs, lifestyle |
| `tech_review` | Spec overlays, data | Product reviews |
| `my_channel` | Custom branding | Your channel (editable) |

## Speed Modes

| Mode | Speed | Quality | Use When |
|------|-------|---------|----------|
| `--superfast` | 4-8x faster | Good | Quick previews, bulk processing |
| `--turbo` | 3-4x faster | Good | Fast turnaround needed |
| `--quality fast` | 2x faster | Decent | Time-sensitive |
| `--quality balanced` | 1x (default) | Great | Normal use |
| `--quality high` | 0.5x | Best | Final export, max quality |

## Platform Specs

| Platform | Resolution | Max Duration | Codec |
|----------|------------|-------------|-------|
| YouTube Shorts | 1080x1920 | 60 seconds | h264, CRF 23, faststart |
| TikTok | 1080x1920 | 180 seconds | h264 baseline, CRF 22 |
| Instagram Reels | 1080x1920 | 90 seconds | h264, CRF 22, yuv420p |

## Project Structure

```
yt-shorts-factory/
├── main.py                      # CLI entry point (Click + interactive menu)
├── setup.py                     # pip-installable package (v4.0)
├── requirements.txt             # Python dependencies
├── .env.example                 # Configuration template
├── .env                         # Your local configuration (gitignored)
├── .gitignore                   # Git ignore rules
├── Makefile                     # make install / run / test / zip
├── README.md                    # This file
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic BaseSettings (100+ validated fields)
│
├── core/
│   ├── __init__.py
│   ├── analyzer.py              # Multi-signal engagement analysis
│   ├── audio_enhancer.py        # Noise reduction, compression, normalization
│   ├── channel_pattern.py       # 10+ branding patterns with hooks, CTA, overlays
│   ├── clip_rater.py            # Clip quality scoring and ranking
│   ├── content_moderator.py     # Content safety checks
│   ├── downloader.py            # yt-dlp video fetcher with retry
│   ├── face_tracker.py          # Face detection for smart crop
│   ├── logo_stamper.py          # Logo overlay with fade-in
│   ├── metadata_generator.py    # Title/description/hashtag generation
│   ├── motion_detector.py       # Motion-based clip selection
│   ├── parallel_pipeline.py     # Parallel executor + SpeedOptimizer
│   ├── pipeline.py              # Master 13-step orchestrator with checkpoints
│   ├── platform_exporter.py     # Platform-specific encoding
│   ├── shorts_converter.py      # Smart crop + reframe 9:16
│   ├── subtitle_engine.py       # ASS subtitle generator (12 animations)
│   ├── superfast.py             # Single-pass FFmpeg pipeline
│   ├── thumbnail_generator.py   # Thumbnail extraction
│   └── transcriber.py           # Whisper AI word timestamps + faster-whisper
│
├── database/
│   ├── __init__.py
│   └── db.py                    # SQLAlchemy ORM (Job, Video models)
│
├── scheduler/
│   ├── __init__.py
│   ├── job_queue.py             # SQLite-backed job queue
│   └── worker.py                # Background worker with auto-scaling
│
├── utils/
│   ├── __init__.py
│   ├── ffmpeg_utils.py          # FFmpeg subprocess wrappers + HW detect
│   ├── file_utils.py            # Safe file operations + cleanup
│   ├── interactive_menu.py      # Arrow-key TUI menu (questionary + rich)
│   ├── logger.py                # Structured rotating logger
│   ├── progress.py              # Rich multi-step progress tracker
│   └── retry.py                 # Exponential backoff decorator
│
├── assets/
│   ├── logo.png                 # Channel logo (replace with yours)
│   └── patterns/
│       └── my_channel.json      # Custom channel branding pattern
│
├── scripts/
│   ├── install_deps.sh          # One-shot dependency installer
│   ├── batch_run.sh             # Batch process URLs
│   └── watch_folder.sh          # Watch folder for URL files
│
├── tests/
│   ├── __init__.py
│   ├── test_ai_metadata.py
│   ├── test_analyzer.py
│   ├── test_clip_rater.py
│   ├── test_downloader.py
│   ├── test_logo_stamper.py
│   ├── test_pipeline.py
│   └── test_subtitle_engine.py
│
└── output/                      # Created at runtime (gitignored)
    ├── downloads/               # Raw downloaded videos
    ├── shorts/                  # Platform-specific outputs
    ├── metadata/                # JSON metadata files
    ├── thumbnails/              # Generated thumbnails
    └── logs/                    # Rotating structured logs
```

## Performance Tips

| Tip | Impact |
|-----|--------|
| Use `--superfast` | 4-8x faster, single FFmpeg pass |
| Use `--turbo` | 3-4x faster, ultrafast + tiny Whisper |
| CUDA GPU (Whisper) | 4-10x faster transcription |
| NVENC GPU (FFmpeg) | 2-5x faster video encoding |
| `pip install faster-whisper` | 4x faster transcription |
| Use `--preset 25s` | Shorter clips = faster processing |
| Use `--no-subs` | Skips Whisper entirely, max speed |
| `--platforms youtube` only | Skip unnecessary platform exports |

### Whisper Model Comparison

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | ~1 GB | 10x | Basic |
| base | ~1 GB | 6x | Good |
| small | ~2 GB | 3x | Very good |
| medium | ~5 GB | 1.5x | Excellent |
| large | ~10 GB | 1x | Best |

## Configuration

All settings are configurable via `.env`. Copy `.env.example` and customize:

```bash
cp .env.example .env
```

Key settings: `WHISPER_MODEL`, `WHISPER_DEVICE`, `FFMPEG_HW_ACCEL`, `LOGO_PATH`, `CHANNEL_PATTERN`, `CLIP_DURATION_PRESET`.

Full reference in the `.env.example` file.

## License

MIT License — use freely in personal and commercial projects.

---

**Developed by Abrar Hussain**
