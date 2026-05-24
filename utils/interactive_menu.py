"""
utils/interactive_menu.py — Beautiful interactive TUI menu for yt-shorts-factory.

Provides arrow-key navigable menus with animated highlight, color-coded
categories, and step-by-step guided workflow. No need to memorize CLI flags.

Usage:
    python main.py            # Launches interactive menu
    python main.py menu       # Explicit interactive menu command

Dependencies: questionary (arrow-key menus), rich (rendering)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# ── Try importing questionary for interactive menus ─────────
try:
    import questionary
    from questionary import Style
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

# ── Custom questionary style ─────────────────────────────────
CUSTOM_STYLE = Style([
    ('qmark', 'fg:#673ab7 bold'),        # Question mark
    ('question', 'bold'),                 # Question text
    ('answer', 'fg:#f44336 bold'),        # Selected answer
    ('pointer', 'fg:#673ab7 bold'),       # Selection pointer
    ('highlighted', 'fg:#673ab7 bold'),   # Highlighted choice
    ('selected', 'fg:#cc5454'),           # Selected checkbox
    ('separator', 'fg:#cc5454'),          # Separator
    ('instruction', ''),                  # Instruction text
    ('text', ''),                         # Regular text
]) if HAS_QUESTIONARY else None


# ═══════════════════════════════════════════════════════════
#  Banner
# ═══════════════════════════════════════════════════════════

_BANNER = r"""
[yellow]   __   _____  __    ____  __  ____  ____
  / /  / ___/ / /   / __/ / / / __/ / __/
 / /__/ /__  / /__ / /__ / /_/ /__ / /__
/____/\___/ /____//___//___//___//____/[/yellow]

[cyan]  YouTube Shorts Factory v4.0[/cyan]
[dim]  Interactive Mode — Use arrow keys to navigate[/dim]
[bold magenta]  Developed by Abrar Hussain[/bold magenta]
"""


def _print_interactive_banner() -> None:
    """Print the interactive mode banner."""
    console.print(_BANNER)
    console.print("[dim]  Use UP/DOWN arrows to move, ENTER to select, ESC to go back[/dim]")
    console.print()


# ═══════════════════════════════════════════════════════════
#  Menu Items Data
# ═══════════════════════════════════════════════════════════

# Main menu options
MAIN_MENU = [
    ("Create Shorts", "craft", "Process a video into shorts (step-by-step)"),
    ("Quick Shorts", "quick", "Superfast mode — just paste URL and go!"),
    ("Batch Process", "batch", "Process multiple URLs at once"),
    ("View History", "history", "Show recent pipeline job history"),
    ("View Stats", "stats", "Show aggregate pipeline statistics"),
    ("Queue Job", "queue", "Add URLs to the job queue"),
    ("Worker Status", "worker", "Manage the background worker"),
    ("Configuration", "config", "Display or edit configuration"),
    ("Verify Setup", "verify", "Check all system dependencies"),
    ("Duration Presets", "presets", "Show available clip duration presets"),
    ("Channel Patterns", "patterns", "Show available channel branding patterns"),
    ("Cleanup", "cleanup", "Clean up old files and database"),
    ("About", "about", "Developer info & project credits"),
    ("GPU Status", "gpu", "Check GPU acceleration (CUDA + NVENC)"),
    ("Exit", "exit", "Quit the application"),
]

# Duration options
DURATION_OPTIONS = [
    ("25 seconds (Quick Hook)", "25", "Best for TikTok fast scroll, attention grabbers"),
    ("45 seconds (Standard)", "45", "Sweet spot for YouTube Shorts & Reels"),
    ("60 seconds (1 minute)", "60", "Extended hook with more context"),
    ("90 seconds (1.5 min)", "90", "Story-driven shorts"),
    ("180 seconds (3 min)", "180", "Long-form Shorts, full storytelling"),
    ("Custom duration...", "custom", "Enter a custom duration in seconds"),
]

# Speed mode options
SPEED_OPTIONS = [
    ("Superfast", "superfast", "Single-pass FFmpeg, center crop, minimal analysis (4-8x faster)"),
    ("Turbo", "turbo", "Ultrafast encoding, tiny whisper, skip extras (3-4x faster)"),
    ("Fast", "fast", "Fast encoding, no face tracking, basic analysis"),
    ("Balanced", "balanced", "Good balance of quality and speed (default)"),
    ("High Quality", "high", "Best quality, slower processing"),
]

# Platform options
PLATFORM_OPTIONS = [
    ("YouTube", "youtube", "YouTube Shorts format"),
    ("TikTok", "tiktok", "TikTok format"),
    ("Instagram Reels", "reels", "Instagram Reels format"),
    ("All Platforms", "all", "Export for all platforms"),
]

# Aspect ratio options
ASPECT_OPTIONS = [
    ("9:16 (Vertical)", "9:16", "Standard vertical video (1080x1920)"),
    ("1:1 (Square)", "1:1", "Square format (1080x1080)"),
    ("4:5 (Portrait)", "4:5", "Instagram portrait (1080x1350)"),
]

# Animation options
ANIMATION_OPTIONS = [
    ("Karaoke (word highlight)", "karaoke", "Highlights each word as it's spoken"),
    ("Fade In/Out", "fade", "Smooth fade transitions"),
    ("Pop", "pop", "Words pop in with emphasis"),
    ("Glow", "glow", "Neon glow effect on words"),
    ("Typewriter", "typewriter", "Characters appear one by one"),
    ("Bounce", "bounce", "Words bounce into view"),
    ("Wave", "wave", "Wave animation across text"),
    ("None", "none", "No animation (fastest)"),
]

# Channel pattern options
PATTERN_OPTIONS = [
    ("Viral Hype", "viral_hype", "High-energy viral content style"),
    ("Chill Vibes", "chill_vibes", "Relaxed, ambient content style"),
    ("News Alert", "news_alert", "Breaking news / urgent style"),
    ("Educational", "educational", "Tutorial / explainer style"),
    ("Gaming Clips", "gaming_clips", "Gaming highlights style"),
    ("Motivational", "motivational", "Inspirational quotes / talks"),
    ("Comedy Clip", "comedy_clip", "Funny moments style"),
    ("Lifestyle", "lifestyle", "Vlog / lifestyle content"),
    ("Tech Review", "tech_review", "Product review style"),
    ("My Channel", "my_channel", "Your personal channel branding"),
    ("No Pattern", "none", "No channel pattern applied"),
]


# ═══════════════════════════════════════════════════════════
#  Interactive Menu Helpers
# ═══════════════════════════════════════════════════════════

def _select(
    title: str,
    choices: list[tuple[str, str, str]],
    instruction: str = "Use arrows to move, ENTER to select",
) -> Optional[str]:
    """Show an interactive selection menu with arrow-key navigation.

    Args:
        title: Menu title/question text.
        choices: List of (display_name, value, description) tuples.
        instruction: Helper text shown below the menu.

    Returns:
        Selected value string, or None if cancelled.
    """
    if not HAS_QUESTIONARY:
        # Fallback: simple numbered menu
        return _fallback_select(title, choices)

    # Build questionary choices with descriptions
    q_choices = []
    for display, value, desc in choices:
        if desc:
            q_choices.append(f"{display}  [dim]{desc}[/dim]")
        else:
            q_choices.append(display)

    # Map display text back to values
    display_to_value = {}
    for (display, value, _), q_choice in zip(choices, q_choices):
        display_to_value[q_choice] = value

    try:
        answer = questionary.select(
            title,
            choices=q_choices,
            style=CUSTOM_STYLE,
            instruction=instruction,
        ).ask()

        if answer is None:
            return None
        return display_to_value.get(answer, answer)
    except (KeyboardInterrupt, EOFError):
        return None


def _checkbox(
    title: str,
    choices: list[tuple[str, str, str]],
) -> list[str]:
    """Show an interactive checkbox menu for multi-select.

    Args:
        title: Menu title/question text.
        choices: List of (display_name, value, description) tuples.

    Returns:
        List of selected value strings.
    """
    if not HAS_QUESTIONARY:
        return _fallback_checkbox(title, choices)

    q_choices = []
    display_to_value = {}
    for display, value, desc in choices:
        q_choice = f"{display}  [dim]{desc}[/dim]" if desc else display
        q_choices.append(q_choice)
        display_to_value[q_choice] = value

    try:
        answers = questionary.checkbox(
            title,
            choices=q_choices,
            style=CUSTOM_STYLE,
        ).ask()

        if answers is None:
            return []
        return [display_to_value.get(a, a) for a in answers]
    except (KeyboardInterrupt, EOFError):
        return []


def _text_input(
    title: str,
    default: str = "",
) -> Optional[str]:
    """Show an interactive text input prompt.

    Args:
        title: Prompt text.
        default: Default value.

    Returns:
        Entered text, or None if cancelled.
    """
    if not HAS_QUESTIONARY:
        return _fallback_input(title, default)

    try:
        answer = questionary.text(
            title,
            default=default,
            style=CUSTOM_STYLE,
        ).ask()
        return answer
    except (KeyboardInterrupt, EOFError):
        return None


def _confirm(title: str, default: bool = True) -> bool:
    """Show an interactive yes/no confirmation.

    Args:
        title: Question text.
        default: Default answer.

    Returns:
        True or False.
    """
    if not HAS_QUESTIONARY:
        return _fallback_confirm(title, default)

    try:
        return questionary.confirm(
            title,
            default=default,
            style=CUSTOM_STYLE,
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return False


# ═══════════════════════════════════════════════════════════
#  Fallback (no questionary) implementations
# ═══════════════════════════════════════════════════════════

def _fallback_select(
    title: str,
    choices: list[tuple[str, str, str]],
) -> Optional[str]:
    """Numbered fallback menu when questionary is not available."""
    console.print(f"\n[bold]{title}[/bold]\n")
    for i, (display, _, desc) in enumerate(choices, 1):
        if desc:
            console.print(f"  [cyan]{i:2d}.[/cyan] {display}  [dim]{desc}[/dim]")
        else:
            console.print(f"  [cyan]{i:2d}.[/cyan] {display}")

    console.print()
    try:
        choice = input("Enter number: ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(choices):
            return choices[idx][1]
        console.print("[red]Invalid selection[/red]")
        return None
    except (ValueError, KeyboardInterrupt, EOFError):
        return None


def _fallback_checkbox(
    title: str,
    choices: list[tuple[str, str, str]],
) -> list[str]:
    """Numbered fallback checkbox when questionary is not available."""
    console.print(f"\n[bold]{title}[/bold]  (comma-separated numbers)\n")
    for i, (display, _, desc) in enumerate(choices, 1):
        if desc:
            console.print(f"  [cyan]{i:2d}.[/cyan] {display}  [dim]{desc}[/dim]")
        else:
            console.print(f"  [cyan]{i:2d}.[/cyan] {display}")

    console.print()
    try:
        raw = input("Enter numbers (e.g. 1,3,5): ").strip()
        selected = []
        for part in raw.split(","):
            idx = int(part.strip()) - 1
            if 0 <= idx < len(choices):
                selected.append(choices[idx][1])
        return selected
    except (ValueError, KeyboardInterrupt, EOFError):
        return []


def _fallback_input(title: str, default: str = "") -> Optional[str]:
    """Fallback text input when questionary is not available."""
    prompt = f"{title}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    try:
        val = input(prompt).strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        return None


def _fallback_confirm(title: str, default: bool = True) -> bool:
    """Fallback confirmation when questionary is not available."""
    hint = "Y/n" if default else "y/N"
    try:
        val = input(f"{title} [{hint}]: ").strip().lower()
        if not val:
            return default
        return val in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False


# ═══════════════════════════════════════════════════════════
#  Step-by-Step Shorts Creator
# ═══════════════════════════════════════════════════════════

def interactive_craft_shorts() -> None:
    """Guided step-by-step shorts creation with interactive menus.

    Walks the user through:
    1. Enter video URL
    2. Select duration
    3. Choose speed/quality mode
    4. Select target platforms
    5. Choose aspect ratio
    6. Select subtitle animation
    7. Optional channel pattern
    8. Confirm and run
    """
    _print_interactive_banner()
    console.print(Panel(
        "[bold]Step-by-Step Shorts Creator[/bold]\n"
        "[dim]Answer each question to configure your shorts. Press ESC to go back.[/dim]",
        border_style="cyan",
    ))

    # ── Step 1: URL ──────────────────────────────────────────
    console.print("\n[bold cyan]Step 1:[/bold cyan] Enter Video URL")
    url = _text_input("YouTube video URL:", default="")
    if not url:
        console.print("[yellow]No URL provided. Returning to menu.[/yellow]")
        return

    # ── Step 2: Duration ─────────────────────────────────────
    console.print("\n[bold cyan]Step 2:[/bold cyan] Select Clip Duration")
    duration_val = _select(
        "How long should each short be?",
        DURATION_OPTIONS,
    )
    if duration_val is None:
        console.print("[yellow]Cancelled. Returning to menu.[/yellow]")
        return

    custom_duration = None
    if duration_val == "custom":
        custom_val = _text_input("Enter duration in seconds:", default="45")
        if custom_val:
            try:
                custom_duration = int(custom_val)
            except ValueError:
                console.print("[red]Invalid number. Using 45s.[/red]")
                custom_duration = 45
        else:
            custom_duration = 45

    # ── Step 3: Speed Mode ───────────────────────────────────
    console.print("\n[bold cyan]Step 3:[/bold cyan] Choose Speed/Quality Mode")
    speed_val = _select(
        "Select processing speed:",
        SPEED_OPTIONS,
    )
    if speed_val is None:
        speed_val = "balanced"

    # ── Step 4: Platforms ────────────────────────────────────
    console.print("\n[bold cyan]Step 4:[/bold cyan] Select Target Platforms")
    platform_val = _select(
        "Which platforms to export for?",
        PLATFORM_OPTIONS,
    )
    if platform_val is None:
        platform_val = "all"

    # ── Step 5: Aspect Ratio ─────────────────────────────────
    console.print("\n[bold cyan]Step 5:[/bold cyan] Choose Aspect Ratio")
    aspect_val = _select(
        "Select output aspect ratio:",
        ASPECT_OPTIONS,
    )
    if aspect_val is None:
        aspect_val = "9:16"

    # ── Step 6: Subtitle Animation ───────────────────────────
    console.print("\n[bold cyan]Step 6:[/bold cyan] Choose Subtitle Animation")
    anim_val = _select(
        "How should subtitles animate?",
        ANIMATION_OPTIONS,
    )
    if anim_val is None:
        anim_val = "karaoke"

    # ── Step 7: Channel Pattern ──────────────────────────────
    console.print("\n[bold cyan]Step 7:[/bold cyan] Choose Channel Pattern")
    pattern_val = _select(
        "Select a channel branding pattern (or none):",
        PATTERN_OPTIONS,
    )
    if pattern_val is None:
        pattern_val = "none"

    # ── Step 8: Optional Settings ────────────────────────────
    console.print("\n[bold cyan]Step 8:[/bold cyan] Additional Options")

    enhance_audio = _confirm("Enhance audio? (noise reduction, compression, normalization)", default=False)
    blur_bg = _confirm("Blur background instead of crop?", default=False)
    no_logo = not _confirm("Add logo watermark?", default=True)
    no_subs = not _confirm("Add subtitles?", default=True)

    # ── Confirmation ─────────────────────────────────────────
    final_duration = custom_duration if custom_duration else int(duration_val)
    platforms_list = ["youtube", "tiktok", "reels"] if platform_val == "all" else [platform_val]

    # Build summary table
    summary = Table(title="Configuration Summary", show_lines=True)
    summary.add_column("Setting", style="cyan bold")
    summary.add_column("Value", style="green bold")

    summary.add_row("URL", url[:70] + ("..." if len(url) > 70 else ""))
    summary.add_row("Duration", f"{final_duration}s")
    summary.add_row("Speed Mode", speed_val)
    summary.add_row("Platforms", ", ".join(platforms_list))
    summary.add_row("Aspect Ratio", aspect_val)
    summary.add_row("Animation", anim_val)
    summary.add_row("Pattern", pattern_val if pattern_val != "none" else "(none)")
    summary.add_row("Enhance Audio", "Yes" if enhance_audio else "No")
    summary.add_row("Blur Background", "Yes" if blur_bg else "No")
    summary.add_row("Logo", "No" if no_logo else "Yes")
    summary.add_row("Subtitles", "No" if no_subs else "Yes")

    console.print()
    console.print(summary)

    if not _confirm("\nStart processing with these settings?", default=True):
        console.print("[yellow]Cancelled. Returning to menu.[/yellow]")
        return

    # ── Run the pipeline ─────────────────────────────────────
    console.print("\n[bold green]Starting pipeline...[/bold green]\n")

    # Build the CLI arguments and invoke the run command
    from config.settings import get_settings
    settings = get_settings()

    # Apply speed mode settings
    if speed_val == "superfast":
        settings.SUPERFAST_MODE = True
        settings.apply_superfast()
    elif speed_val == "turbo":
        settings.TURBO_MODE = True
        settings.apply_turbo()

    # Apply duration
    settings.CLIP_DURATION = final_duration

    # Apply aspect ratio
    ratio_map = {"9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350)}
    if aspect_val in ratio_map:
        settings.OUTPUT_WIDTH, settings.OUTPUT_HEIGHT = ratio_map[aspect_val]

    # Apply platforms
    settings.EXPORT_YOUTUBE = "youtube" in platforms_list
    settings.EXPORT_TIKTOK = "tiktok" in platforms_list
    settings.EXPORT_REELS = "reels" in platforms_list

    # Apply animation
    settings.SUBTITLE_ANIMATION = anim_val

    # Apply pattern
    if pattern_val != "none":
        settings.CHANNEL_PATTERN = pattern_val

    # Quality mapping
    quality_map = {
        "superfast": "fast",
        "turbo": "fast",
        "fast": "fast",
        "balanced": "balanced",
        "high": "high",
    }
    quality = quality_map.get(speed_val, "balanced")

    quality_settings = {
        "fast": {"FFMPEG_PRESET": "ultrafast", "FFMPEG_CRF": 28},
        "balanced": {"FFMPEG_PRESET": "fast", "FFMPEG_CRF": 23},
        "high": {"FFMPEG_PRESET": "slow", "FFMPEG_CRF": 18},
    }
    if quality in quality_settings:
        for k, v in quality_settings[quality].items():
            setattr(settings, k, v)

    # Run pipeline
    try:
        if speed_val == "superfast":
            from core.superfast import superfast_pipeline
            sf_result = superfast_pipeline(
                url=url,
                duration=final_duration,
                skip_subs=no_subs,
                no_logo=no_logo,
                platforms=platforms_list,
                settings=settings,
                blur_bg=blur_bg,
            )
            if sf_result.success:
                console.print(f"\n[bold green]Short created successfully![/bold green]")
                if sf_result.output_path:
                    console.print(f"  Output: {sf_result.output_path}")
                    console.print(f"  Size: {sf_result.file_size_human}")
                    console.print(f"  Time: {sf_result.duration:.1f}s")
            else:
                console.print(f"\n[bold red]Pipeline failed![/bold red]")
        else:
            from core.pipeline import run_pipeline
            result = run_pipeline(
                url=url,
                duration=final_duration,
                skip_subs=no_subs,
                no_logo=no_logo,
                enhance_audio=enhance_audio,
                blur_background=blur_bg,
                platforms=platforms_list,
                quality=quality,
            )
            if result.success:
                console.print(f"\n[bold green]Short created successfully![/bold green]")
            else:
                console.print(f"\n[bold red]Pipeline failed: {result.error}[/bold red]")
    except Exception as exc:
        console.print(f"\n[bold red]Error: {exc}[/bold red]")

    # Ask what to do next
    console.print()
    if _confirm("Process another video?", default=False):
        interactive_craft_shorts()


def interactive_quick_shorts() -> None:
    """Ultra-fast mode: Just paste a URL and go with sensible defaults.

    Uses superfast mode, 45s duration, all platforms, karaoke subtitles.
    """
    _print_interactive_banner()
    console.print(Panel(
        "[bold magenta]Quick Shorts Mode[/bold magenta]\n"
        "[dim]Fastest way to create shorts! Just paste a URL and we handle the rest.[/dim]\n"
        "[dim]Uses: Superfast mode, 45s duration, all platforms, karaoke subtitles[/dim]",
        border_style="magenta",
    ))

    url = _text_input("Paste YouTube video URL:")
    if not url:
        console.print("[yellow]No URL provided.[/yellow]")
        return

    # Quick duration choice
    duration_val = _select(
        "Clip duration:",
        [
            ("25 seconds", "25", "Quick hook"),
            ("45 seconds (default)", "45", "Standard short"),
            ("60 seconds", "60", "Extended"),
            ("3 minutes", "180", "Long-form"),
        ],
    )
    if duration_val is None:
        duration_val = "45"

    console.print(f"\n[bold magenta]Processing with Superfast mode...[/bold magenta]\n")

    from config.settings import get_settings
    settings = get_settings()
    settings.SUPERFAST_MODE = True
    settings.apply_superfast()
    settings.CLIP_DURATION = int(duration_val)

    try:
        from core.superfast import superfast_pipeline
        sf_result = superfast_pipeline(
            url=url,
            duration=int(duration_val),
            skip_subs=False,
            no_logo=False,
            platforms=["youtube", "tiktok", "reels"],
            settings=settings,
        )
        if sf_result.success:
            console.print(f"\n[bold green]Done![/bold green] Output: {sf_result.output_path}")
            console.print(f"  Size: {sf_result.file_size_human}  Time: {sf_result.duration:.1f}s")
        else:
            console.print(f"\n[bold red]Failed![/bold red]")
    except Exception as exc:
        console.print(f"\n[bold red]Error: {exc}[/bold red]")

    if _confirm("\nProcess another?", default=False):
        interactive_quick_shorts()


# ═══════════════════════════════════════════════════════════
#  Main Interactive Menu Loop
# ═══════════════════════════════════════════════════════════

def interactive_main() -> None:
    """Main interactive menu loop — the primary entry point.

    Displays the main menu with arrow-key navigation and dispatches
    to the appropriate sub-menu or action based on user selection.
    """
    _print_interactive_banner()

    while True:
        choice = _select(
            "What would you like to do?",
            MAIN_MENU,
            instruction="Use arrows + ENTER (ESC to exit)",
        )

        if choice is None or choice == "exit":
            console.print("\n[dim]Goodbye! Happy short-making![/dim]")
            break

        elif choice == "craft":
            interactive_craft_shorts()

        elif choice == "quick":
            interactive_quick_shorts()

        elif choice == "history":
            _run_cli_command("history")

        elif choice == "stats":
            _run_cli_command("stats")

        elif choice == "queue":
            _interactive_queue()

        elif choice == "worker":
            _interactive_worker()

        elif choice == "config":
            _interactive_config()

        elif choice == "verify":
            _run_cli_command("verify")

        elif choice == "presets":
            _run_cli_command("presets")

        elif choice == "patterns":
            _run_cli_command("patterns")

        elif choice == "cleanup":
            _interactive_cleanup()

        elif choice == "batch":
            _interactive_batch()

        elif choice == "about":
            _run_cli_command("about")

        elif choice == "gpu":
            _run_cli_command("gpu")

        # Pause before showing menu again
        console.print()
        if choice not in ("craft", "quick", "batch"):
            input("Press ENTER to continue...")


def _run_cli_command(cmd_name: str, args: list[str] | None = None) -> None:
    """Run a Click CLI command programmatically.

    Args:
        cmd_name: CLI command name.
        args: Optional list of additional arguments.
    """
    from click.testing import CliRunner
    from main import cli

    runner = CliRunner()
    cmd_args = [cmd_name] + (args or [])
    result = runner.invoke(cli, cmd_args)
    if result.output:
        console.print(result.output)
    if result.exception and not isinstance(result.exception, SystemExit):
        console.print(f"[red]Error: {result.exception}[/red]")


def _interactive_queue() -> None:
    """Interactive queue management."""
    console.print(Panel("[bold]Job Queue[/bold]", border_style="cyan"))

    action = _select(
        "What would you like to do?",
        [
            ("Add URL to queue", "add", "Enqueue a new video URL"),
            ("View queue status", "status", "Show pending/running jobs"),
            ("Back to main menu", "back", ""),
        ],
    )

    if action == "add":
        url = _text_input("Enter YouTube URL:")
        if url:
            priority = _select("Priority:", [
                ("High", "high", "Process first"),
                ("Medium", "medium", "Normal priority"),
                ("Low", "low", "Process last"),
            ])
            if priority:
                _run_cli_command("queue", [url, "--priority", priority])
                console.print("[green]Job added to queue![/green]")

    elif action == "status":
        _run_cli_command("worker", ["--status"])


def _interactive_worker() -> None:
    """Interactive worker management."""
    console.print(Panel("[bold]Worker Management[/bold]", border_style="cyan"))

    action = _select(
        "Worker options:",
        [
            ("Start worker", "start", "Start the background worker daemon"),
            ("Show status", "status", "Show worker and queue status"),
            ("Health check", "health", "Show worker health check"),
            ("Recover stale jobs", "recover", "Recover jobs stuck in 'running' state"),
            ("Back to main menu", "back", ""),
        ],
    )

    if action == "start":
        _run_cli_command("worker", ["--start"])
    elif action == "status":
        _run_cli_command("worker", ["--status"])
    elif action == "health":
        _run_cli_command("worker", ["--health"])
    elif action == "recover":
        _run_cli_command("worker", ["--recover"])


def _interactive_config() -> None:
    """Interactive configuration viewer/editor."""
    console.print(Panel("[bold]Configuration[/bold]", border_style="cyan"))

    action = _select(
        "Config options:",
        [
            ("View current config", "view", "Display all settings"),
            ("Change Whisper model", "whisper", "Switch between tiny/base/small/medium/large"),
            ("Change duration preset", "duration", "Switch clip duration preset"),
            ("Reset to defaults", "reset", "Reset .env to defaults"),
            ("Back to main menu", "back", ""),
        ],
    )

    if action == "view":
        _run_cli_command("config")
    elif action == "whisper":
        model = _select("Select Whisper model:", [
            ("tiny (fastest, lowest accuracy)", "tiny", "39M params, ~32x realtime"),
            ("base (fast, good accuracy)", "base", "74M params, ~16x realtime"),
            ("small (balanced)", "small", "244M params, ~6x realtime"),
            ("medium (slow, high accuracy)", "medium", "769M params, ~2x realtime"),
            ("large (slowest, best accuracy)", "large", "1550M params, ~1x realtime"),
        ])
        if model:
            _run_cli_command("config", ["--set", f"WHISPER_MODEL={model}"])
            console.print(f"[green]Whisper model set to: {model}[/green]")
    elif action == "duration":
        dur = _select("Select duration preset:", DURATION_OPTIONS)
        if dur and dur != "custom":
            _run_cli_command("config", ["--set", f"CLIP_DURATION={dur}"])
            console.print(f"[green]Duration set to: {dur}s[/green]")
    elif action == "reset":
        if _confirm("Reset all configuration to defaults?", default=False):
            _run_cli_command("config", ["--reset"])


def _interactive_cleanup() -> None:
    """Interactive cleanup management."""
    console.print(Panel("[bold]Cleanup[/bold]", border_style="cyan"))

    action = _select(
        "Cleanup options:",
        [
            ("Quick cleanup (30+ days)", "quick", "Remove files older than 30 days"),
            ("Deep cleanup (7+ days)", "deep", "Remove files older than 7 days"),
            ("Preview only (dry run)", "preview", "See what would be deleted"),
            ("Back to main menu", "back", ""),
        ],
    )

    if action == "quick":
        _run_cli_command("cleanup", ["--days", "30"])
    elif action == "deep":
        _run_cli_command("cleanup", ["--days", "7"])
    elif action == "preview":
        _run_cli_command("cleanup", ["--dry-run"])


def _interactive_batch() -> None:
    """Interactive batch processing — process multiple URLs at once."""
    _print_interactive_banner()
    console.print(Panel(
        "[bold]Batch Process[/bold]\n"
        "[dim]Process multiple YouTube videos into shorts at once.\n"
        "Enter URLs one by one, then configure settings for all of them.[/dim]",
        border_style="green",
    ))

    # Collect URLs
    urls: list[str] = []
    console.print("\n[bold cyan]Step 1:[/bold cyan] Enter YouTube URLs (one at a time)")
    console.print("[dim]Leave blank and press ENTER when done[/dim]\n")

    while True:
        url = _text_input(f"URL #{len(urls) + 1} (or blank to finish):")
        if not url:
            break
        urls.append(url)
        console.print(f"  [green]Added:[/green] {url[:70]}")

    if not urls:
        console.print("[yellow]No URLs provided. Returning to menu.[/yellow]")
        return

    console.print(f"\n[green]Collected {len(urls)} URL(s)[/green]")

    # Common settings for all URLs
    console.print("\n[bold cyan]Step 2:[/bold cyan] Configure settings for all URLs")

    duration_val = _select("Clip duration:", [
        ("25 seconds", "25", "Quick hook"),
        ("45 seconds (default)", "45", "Standard short"),
        ("60 seconds", "60", "Extended"),
        ("3 minutes", "180", "Long-form"),
    ])
    if duration_val is None:
        duration_val = "45"

    speed_val = _select("Speed mode:", SPEED_OPTIONS)
    if speed_val is None:
        speed_val = "superfast"

    platform_val = _select("Platforms:", PLATFORM_OPTIONS)
    if platform_val is None:
        platform_val = "all"

    # Summary
    platforms_list = ["youtube", "tiktok", "reels"] if platform_val == "all" else [platform_val]

    summary = Table(title="Batch Summary", show_lines=True)
    summary.add_column("Setting", style="cyan bold")
    summary.add_column("Value", style="green bold")
    summary.add_row("URLs", str(len(urls)))
    summary.add_row("Duration", f"{duration_val}s")
    summary.add_row("Speed", speed_val)
    summary.add_row("Platforms", ", ".join(platforms_list))
    console.print()
    console.print(summary)

    if not _confirm("\nStart batch processing?", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # Process each URL
    console.print(f"\n[bold green]Processing {len(urls)} URL(s)...[/bold green]\n")

    from config.settings import get_settings
    settings = get_settings()

    if speed_val == "superfast":
        settings.SUPERFAST_MODE = True
        settings.apply_superfast()
    elif speed_val == "turbo":
        settings.TURBO_MODE = True
        settings.apply_turbo()

    settings.CLIP_DURATION = int(duration_val)
    settings.EXPORT_YOUTUBE = "youtube" in platforms_list
    settings.EXPORT_TIKTOK = "tiktok" in platforms_list
    settings.EXPORT_REELS = "reels" in platforms_list

    results: list[tuple[str, bool]] = []
    for i, url in enumerate(urls, 1):
        console.print(f"[bold]Processing {i}/{len(urls)}:[/bold] {url[:60]}")
        try:
            if speed_val in ("superfast", "turbo"):
                from core.superfast import superfast_pipeline
                sf_result = superfast_pipeline(
                    url=url,
                    duration=int(duration_val),
                    skip_subs=False,
                    no_logo=False,
                    platforms=platforms_list,
                    settings=settings,
                )
                results.append((url, sf_result.success))
                if sf_result.success:
                    console.print(f"  [green]OK[/green] - {sf_result.file_size_human} in {sf_result.duration:.1f}s")
                else:
                    console.print(f"  [red]FAILED[/red]")
            else:
                from core.pipeline import run_pipeline
                quality_map = {"fast": "fast", "balanced": "balanced", "high": "high"}
                quality = quality_map.get(speed_val, "balanced")
                result = run_pipeline(
                    url=url,
                    duration=int(duration_val),
                    platforms=platforms_list,
                    quality=quality,
                )
                results.append((url, result.success))
                if result.success:
                    console.print(f"  [green]OK[/green] - {result.total_duration_seconds:.1f}s")
                else:
                    console.print(f"  [red]FAILED[/red] - {result.error[:50]}")
        except Exception as exc:
            console.print(f"  [red]ERROR:[/red] {exc}")
            results.append((url, False))

    # Summary
    success_count = sum(1 for _, ok in results if ok)
    console.print(f"\n[bold]Batch Complete:[/bold] {success_count}/{len(results)} successful")

    if _confirm("\nProcess more URLs?", default=False):
        _interactive_batch()
