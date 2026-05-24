"""
utils/progress.py — Rich multi-step progress tracker for the pipeline.

Provides:
- PipelineProgress: Multi-step pipeline display with live Rich rendering
- StepETACalculator: ETA calculation based on historical step durations
- ProgressExporter: Export progress to JSON for external monitoring
- ConcurrentProgress: Multiple concurrent progress bars
- Spinner: Animated spinner variations for indeterminate progress
- ProgressCallback: Integration callbacks for external systems
- Color themes for different output contexts
- Hierarchical step tree view
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

console = Console()


# ══════════════════════════════════════════════════════════
#  Data Classes
# ══════════════════════════════════════════════════════════

@dataclass
class StepInfo:
    """Internal tracking state for a single pipeline step."""

    name: str
    status: str = "pending"  # pending | running | done | failed | skipped
    duration: float = 0.0
    start_time: float = 0.0
    error: str = ""
    detail: str = ""
    progress: float = 0.0  # 0.0-1.0 sub-progress within the step
    eta_seconds: float = 0.0
    parent: str = ""  # Parent step name for hierarchical display
    can_parallel: bool = False  # Whether this step can run in parallel


@dataclass
class StepDuration:
    """Historical duration record for a step."""

    step_name: str
    duration: float
    timestamp: float


# ══════════════════════════════════════════════════════════
#  Color Themes
# ══════════════════════════════════════════════════════════

class ColorTheme:
    """Color theme configuration for progress display.

    Defines colors for different UI elements and status states.
    Supports multiple presets for different output contexts.
    """

    def __init__(
        self,
        name: str = "default",
        pending: str = "dim",
        running: str = "yellow bold",
        done: str = "green",
        failed: str = "red",
        skipped: str = "dim",
        header: str = "bold",
        border: str = "blue",
        progress_bar: str = "cyan",
        eta: str = "bright_blue",
        detail: str = "dim",
    ) -> None:
        """Initialise the color theme.

        Args:
            name: Theme name identifier.
            pending: Rich style for pending status.
            running: Rich style for running status.
            done: Rich style for done status.
            failed: Rich style for failed status.
            skipped: Rich style for skipped status.
            header: Rich style for header text.
            border: Rich style for border lines.
            progress_bar: Rich style for progress bars.
            eta: Rich style for ETA text.
            detail: Rich style for detail text.
        """
        self.name = name
        self.pending = pending
        self.running = running
        self.done = done
        self.failed = failed
        self.skipped = skipped
        self.header = header
        self.border = border
        self.progress_bar = progress_bar
        self.eta = eta
        self.detail = detail


# Pre-built themes
THEME_DEFAULT = ColorTheme(name="default")
THEME_DARK = ColorTheme(
    name="dark",
    pending="grey50",
    running="bright_yellow",
    done="bright_green",
    failed="bright_red",
    skipped="grey37",
    header="bright_white",
    border="bright_blue",
    progress_bar="bright_cyan",
    eta="bright_magenta",
    detail="grey70",
)
THEME_MINIMAL = ColorTheme(
    name="minimal",
    pending="",
    running="bold",
    done="",
    failed="bold red",
    skipped="dim",
    header="",
    border="",
    progress_bar="",
    eta="",
    detail="dim",
)
THEME_VERBOSE = ColorTheme(
    name="verbose",
    pending="dim cyan",
    running="bold yellow",
    done="bold green",
    failed="bold red on white",
    skipped="dim italic",
    header="bold magenta",
    border="cyan",
    progress_bar="green on black",
    eta="bright_cyan",
    detail="bright_white",
)


def _get_theme_from_env() -> ColorTheme:
    """Select a color theme based on the PROGRESS_THEME environment variable.

    Returns:
        ColorTheme instance matching the env var, or THEME_DEFAULT.
    """
    theme_name = os.environ.get("PROGRESS_THEME", "default").strip().lower()
    themes: dict[str, ColorTheme] = {
        "default": THEME_DEFAULT,
        "dark": THEME_DARK,
        "minimal": THEME_MINIMAL,
        "verbose": THEME_VERBOSE,
    }
    return themes.get(theme_name, THEME_DEFAULT)


# ══════════════════════════════════════════════════════════
#  Status Icons
# ══════════════════════════════════════════════════════════

_STATUS_ICONS = {
    "pending": "[dim]...[/dim]",
    "running": "[yellow bold]>>[/yellow bold]",
    "done": "[green]OK[/green]",
    "failed": "[red]X[/red]",
    "skipped": "[dim]--[/dim]",
}


# ══════════════════════════════════════════════════════════
#  Step ETA Calculator
# ══════════════════════════════════════════════════════════

class StepETACalculator:
    """Calculate ETAs for pipeline steps based on historical data.

    Maintains a rolling window of past step durations and uses the
    average to predict how long future steps will take. Data can be
    persisted to a JSON file for cross-run learning.

    Example:
        eta_calc = StepETACalculator(history_file=Path("eta_history.json"))
        eta_calc.record_step("transcription", 12.5)
        eta = eta_calc.get_eta("encoding")  # Based on past averages
    """

    def __init__(
        self,
        history_file: Path | None = None,
        max_history: int = 100,
    ) -> None:
        """Initialise the ETA calculator.

        Args:
            history_file: Optional path to persist history between runs.
            max_history: Maximum number of historical records per step.
        """
        self.history_file = history_file
        self.max_history = max_history
        self._history: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

        # Load persisted history
        if self.history_file and self.history_file.exists():
            self._load_history()

    def record_step(self, step_name: str, duration: float) -> None:
        """Record a completed step's duration.

        Args:
            step_name: Name of the step.
            duration: Duration in seconds.
        """
        if duration <= 0:
            return

        with self._lock:
            self._history[step_name].append(duration)
            # Keep only the most recent records
            if len(self._history[step_name]) > self.max_history:
                self._history[step_name] = self._history[step_name][-self.max_history:]

        # Persist after recording
        if self.history_file:
            self._save_history()

    def get_eta(self, step_name: str) -> float:
        """Get estimated duration for a step based on historical data.

        Args:
            step_name: Name of the step to estimate.

        Returns:
            Estimated duration in seconds, or 0.0 if no history exists.
        """
        with self._lock:
            durations = self._history.get(step_name, [])
            if not durations:
                return 0.0
            # Use recent average (last 10 records) for better prediction
            recent = durations[-10:]
            return sum(recent) / len(recent)

    def get_total_eta(self, step_names: list[str]) -> float:
        """Get estimated total duration for a list of steps.

        Steps that can run in parallel are counted only once (using
        the maximum duration in the parallel group).

        Args:
            step_names: Ordered list of step names.

        Returns:
            Estimated total duration in seconds.
        """
        total = 0.0
        for step_name in step_names:
            eta = self.get_eta(step_name)
            total += eta
        return total

    def format_eta(self, seconds: float) -> str:
        """Format an ETA value as a human-readable string.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted string like '2m 30s', '1h 5m', or 'N/A'.
        """
        if seconds <= 0:
            return "N/A"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def _save_history(self) -> None:
        """Persist history to the JSON file."""
        if not self.history_file:
            return

        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            data = dict(self._history)
            self.history_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            # Best effort persistence
            pass

    def _load_history(self) -> None:
        """Load history from the JSON file."""
        if not self.history_file or not self.history_file.exists():
            return

        try:
            data = json.loads(self.history_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, list):
                        self._history[key] = [float(v) for v in values if isinstance(v, (int, float))]
        except (json.JSONDecodeError, OSError, ValueError):
            pass


# ══════════════════════════════════════════════════════════
#  Progress Exporter
# ══════════════════════════════════════════════════════════

class ProgressExporter:
    """Export progress state to JSON for external monitoring.

    Serialises the current pipeline progress into a JSON structure
    suitable for consumption by external tools, dashboards, or APIs.

    Example:
        exporter = ProgressExporter(progress_instance)
        exporter.export_to_file(Path("progress.json"))
        json_str = exporter.export_to_string()
    """

    def __init__(self, pipeline: PipelineProgress) -> None:
        """Initialise the exporter with a pipeline reference.

        Args:
            pipeline: PipelineProgress instance to export state from.
        """
        self.pipeline = pipeline

    def export_to_dict(self) -> dict[str, Any]:
        """Export current progress state as a dictionary.

        Returns:
            Dictionary with 'title', 'total_elapsed', 'steps', and
            'summary' keys.
        """
        elapsed = time.time() - self.pipeline._start_time if self.pipeline._start_time else 0

        steps_data: list[dict[str, Any]] = []
        for name in self.pipeline.step_order:
            step = self.pipeline.step_map[name]
            steps_data.append({
                "name": step.name,
                "status": step.status,
                "duration": round(step.duration, 2),
                "progress": round(step.progress, 3),
                "eta_seconds": round(step.eta_seconds, 1),
                "error": step.error,
                "detail": step.detail,
            })

        done_count = sum(1 for s in self.pipeline.step_map.values() if s.status == "done")
        failed_count = sum(1 for s in self.pipeline.step_map.values() if s.status == "failed")
        total_count = len(self.pipeline.step_order)

        return {
            "title": self.pipeline.title,
            "total_elapsed_seconds": round(elapsed, 2),
            "overall_progress": round(done_count / total_count, 3) if total_count > 0 else 0.0,
            "steps": steps_data,
            "summary": {
                "total_steps": total_count,
                "completed": done_count,
                "failed": failed_count,
                "pending": total_count - done_count - failed_count,
            },
            "ffmpeg": {
                "percent": self.pipeline.ffmpeg_percent,
                "speed": self.pipeline.ffmpeg_speed,
                "eta": self.pipeline.ffmpeg_eta,
            } if self.pipeline.show_ffmpeg else None,
        }

    def export_to_string(self) -> str:
        """Export current progress state as a JSON string.

        Returns:
            Formatted JSON string.
        """
        return json.dumps(self.export_to_dict(), indent=2, ensure_ascii=False)

    def export_to_file(self, path: Path) -> None:
        """Export current progress state to a JSON file.

        Args:
            path: File path to write the JSON export to.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.export_to_string(), encoding="utf-8")
        except OSError as exc:
            pass  # Best effort


# ══════════════════════════════════════════════════════════
#  Progress Callbacks
# ══════════════════════════════════════════════════════════

class ProgressCallback:
    """Integration callbacks for external progress monitoring.

    Allows registering callback functions that are invoked whenever
    progress state changes. Useful for integration with websockets,
    message queues, or monitoring systems.

    Example:
        callback = ProgressCallback()
        callback.on_step_start(lambda name: print(f"Started: {name}"))
        callback.on_step_done(lambda name, dur: print(f"Done: {name} ({dur:.1f}s)"))
    """

    def __init__(self) -> None:
        """Initialise the callback registry."""
        self._step_start_callbacks: list[Callable[[str], None]] = []
        self._step_done_callbacks: list[Callable[[str, float], None]] = []
        self._step_failed_callbacks: list[Callable[[str, str], None]] = []
        self._progress_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._lock = threading.Lock()

    def on_step_start(self, callback: Callable[[str], None]) -> None:
        """Register a callback for when a step starts.

        Args:
            callback: Callable that receives the step name.
        """
        with self._lock:
            self._step_start_callbacks.append(callback)

    def on_step_done(self, callback: Callable[[str, float], None]) -> None:
        """Register a callback for when a step completes.

        Args:
            callback: Callable that receives (step_name, duration_seconds).
        """
        with self._lock:
            self._step_done_callbacks.append(callback)

    def on_step_failed(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback for when a step fails.

        Args:
            callback: Callable that receives (step_name, error_message).
        """
        with self._lock:
            self._step_failed_callbacks.append(callback)

    def on_progress_update(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback for general progress updates.

        Args:
            callback: Callable that receives the full progress dictionary.
        """
        with self._lock:
            self._progress_callbacks.append(callback)

    def notify_step_start(self, step_name: str) -> None:
        """Invoke all step-start callbacks.

        Args:
            step_name: Name of the started step.
        """
        for cb in self._step_start_callbacks:
            try:
                cb(step_name)
            except Exception:
                pass

    def notify_step_done(self, step_name: str, duration: float) -> None:
        """Invoke all step-done callbacks.

        Args:
            step_name: Name of the completed step.
            duration: Duration in seconds.
        """
        for cb in self._step_done_callbacks:
            try:
                cb(step_name, duration)
            except Exception:
                pass

    def notify_step_failed(self, step_name: str, error: str) -> None:
        """Invoke all step-failed callbacks.

        Args:
            step_name: Name of the failed step.
            error: Error message.
        """
        for cb in self._step_failed_callbacks:
            try:
                cb(step_name, error)
            except Exception:
                pass

    def notify_progress_update(self, progress_data: dict[str, Any]) -> None:
        """Invoke all progress-update callbacks.

        Args:
            progress_data: Full progress state dictionary.
        """
        for cb in self._progress_callbacks:
            try:
                cb(progress_data)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════
#  Spinner
# ══════════════════════════════════════════════════════════

class Spinner:
    """Animated spinner for indeterminate progress.

    Provides several spinner character sequences that cycle to create
    an animation effect. Can be used independently or as part of
    PipelineProgress.

    Example:
        spinner = Spinner(style="dots")
        for frame in spinner:
            print(f"\\r{frame}", end="", flush=True)
            time.sleep(0.1)
    """

    SPINNERS: dict[str, list[str]] = {
        "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "line": ["-", "\\", "|", "/"],
        "arrow": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
        "braille": ["⡀", "⡁", "⡂", "⡃", "⡄", "⡅", "⡆", "⡇", "⢀", "⢁"],
        "classic": [".", "o", "O", "°", "O", "o"],
        "triangle": ["▸", "▹", "►", "▻"],
        "clock": ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"],
    }

    def __init__(self, style: str = "dots", interval: float = 0.1) -> None:
        """Initialise the spinner.

        Args:
            style: Spinner style name (dots, line, arrow, braille, classic, triangle, clock).
            interval: Seconds between frame changes (default 0.1).

        Raises:
            ValueError: If the style name is not recognised.
        """
        if style not in self.SPINNERS:
            raise ValueError(
                f"Unknown spinner style '{style}'. Available: {list(self.SPINNERS.keys())}"
            )
        self.frames = self.SPINNERS[style]
        self.interval = interval
        self._index = 0
        self._last_update = 0.0

    def current_frame(self) -> str:
        """Get the current spinner frame.

        Returns:
            Current spinner character.
        """
        now = time.time()
        if now - self._last_update >= self.interval:
            self._index = (self._index + 1) % len(self.frames)
            self._last_update = now
        return self.frames[self._index]

    def reset(self) -> None:
        """Reset the spinner to the first frame."""
        self._index = 0
        self._last_update = 0.0


# ══════════════════════════════════════════════════════════
#  Concurrent Progress
# ══════════════════════════════════════════════════════════

class ConcurrentProgress:
    """Multiple concurrent progress bars for parallel operations.

    Uses Rich's Progress widget to display multiple progress bars
    simultaneously, each tracking a different concurrent operation.

    Example:
        cp = ConcurrentProgress()
        bar1 = cp.add_task("Download", total=100)
        bar2 = cp.add_task("Transcode", total=200)
        cp.start()
        cp.update(bar1, advance=50)
        cp.update(bar2, advance=100)
        cp.stop()
    """

    def __init__(self, title: str = "Concurrent Operations") -> None:
        """Initialise the concurrent progress display.

        Args:
            title: Title for the progress display.
        """
        self.title = title
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TimeElapsedColumn(),
            console=console,
        )
        self._live: Live | None = None
        self._tasks: dict[str, int] = {}

    def add_task(self, name: str, total: float = 100.0) -> str:
        """Add a new progress bar.

        Args:
            name: Description for the progress bar.
            total: Total value for 100% completion.

        Returns:
            Task ID string for future updates.
        """
        task_id = self._progress.add_task(name, total=total)
        self._tasks[name] = task_id
        return name

    def update(self, task_name: str, advance: float = 0, completed: float | None = None) -> None:
        """Update a progress bar.

        Args:
            task_name: Name of the task to update.
            advance: Amount to advance the progress.
            completed: Set the absolute completed value.
        """
        if task_name in self._tasks:
            task_id = self._tasks[task_name]
            if completed is not None:
                self._progress.update(task_id, completed=completed)
            else:
                self._progress.advance(task_id, advance)

    def start(self) -> None:
        """Start the live progress display."""
        self._progress.start()

    def stop(self) -> None:
        """Stop the live progress display."""
        self._progress.stop()


# ══════════════════════════════════════════════════════════
#  PipelineProgress — Main Class
# ══════════════════════════════════════════════════════════

class PipelineProgress:
    """Multi-step pipeline progress display using Rich Live rendering.

    Renders an animated display showing:
    - A header panel with pipeline title
    - A vertical step list with status icons and elapsed times
    - An optional FFmpeg sub-progress bar
    - A final summary panel with all stats
    - ETA predictions based on historical data
    - Hierarchical tree view for nested steps
    - Progress export for external monitoring
    - Callback support for integration with external systems
    - Color theming for different output contexts
    """

    def __init__(
        self,
        title: str,
        steps: list[str],
        theme: ColorTheme | None = None,
        eta_calculator: StepETACalculator | None = None,
        callback: ProgressCallback | None = None,
    ) -> None:
        """Initialise the progress tracker.

        Args:
            title: Pipeline title shown in the header.
            steps: Ordered list of step names.
            theme: Color theme for rendering. Defaults to env-based selection.
            eta_calculator: Optional ETA calculator for predictions.
            callback: Optional callback for external integrations.
        """
        self.title = title
        self.theme = theme or _get_theme_from_env()
        self.eta_calculator = eta_calculator
        self.callback = callback or ProgressCallback()
        self.exporter = ProgressExporter(self)

        self.step_map: dict[str, StepInfo] = {
            name: StepInfo(name=name) for name in steps
        }
        self.step_order: list[str] = list(steps)
        self.ffmpeg_percent: float = 0.0
        self.ffmpeg_speed: str = ""
        self.ffmpeg_eta: str = ""
        self.show_ffmpeg: bool = False
        self._start_time: float = 0.0
        self._live: Live | None = None

    def _render(self) -> Panel:
        """Build the current Rich renderable for the Live display."""
        theme = self.theme

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("icon", width=4)
        table.add_column("step", style="bold")
        table.add_column("detail")

        for name in self.step_order:
            step = self.step_map[name]
            icon = _STATUS_ICONS.get(step.status, "?")
            detail = ""

            if step.status == "running":
                elapsed = time.time() - step.start_time
                detail = f"[{theme.running}]{elapsed:.1f}s[/{theme.running}]"

                # Show sub-progress if available
                if step.progress > 0:
                    bar_width = 20
                    filled = int(bar_width * step.progress)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    detail += f" [{theme.progress_bar}][{bar}][/{theme.progress_bar}] {step.progress * 100:.0f}%"

                # Show ETA if available
                if step.eta_seconds > 0:
                    eta_str = self.eta_calculator.format_eta(step.eta_seconds) if self.eta_calculator else f"{step.eta_seconds:.0f}s"
                    detail += f" [{theme.eta}]ETA: {eta_str}[/{theme.eta}]"

            elif step.status == "done":
                detail = f"[{theme.done}]{step.duration:.1f}s[/{theme.done}]"
                if step.detail:
                    detail += f" [{theme.detail}]({step.detail})[/{theme.detail}]"
            elif step.status == "failed":
                detail = f"[{theme.failed}]{step.error}[/{theme.failed}]"
            elif step.status == "skipped":
                detail = f"[{theme.skipped}]skipped[/{theme.skipped}]"

            table.add_row(icon, step.name, detail)

        # FFmpeg sub-progress
        if self.show_ffmpeg:
            bar_width = 30
            filled = int(bar_width * self.ffmpeg_percent / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            ffmpeg_text = (
                f"FFmpeg: [{theme.progress_bar}][{bar}][/{theme.progress_bar}] "
                f"{self.ffmpeg_percent:.1f}%  "
                f"Speed: {self.ffmpeg_speed}  ETA: {self.ffmpeg_eta}"
            )
            table.add_row("", Text(ffmpeg_text), "")

        elapsed_total = time.time() - self._start_time if self._start_time else 0

        # Overall ETA
        eta_text = ""
        if self.eta_calculator:
            remaining_steps = [
                name for name in self.step_order
                if self.step_map[name].status in ("pending", "running")
            ]
            total_remaining = self.eta_calculator.get_total_eta(remaining_steps)
            if total_remaining > 0:
                eta_text = f"  [{theme.eta}]ETA: {self.eta_calculator.format_eta(total_remaining)}[/{theme.eta}]"

        header = f"[{theme.header}]{self.title}[/{theme.header}]  ({elapsed_total:.1f}s){eta_text}"
        return Panel(table, title=header, border_style=theme.border, padding=(1, 2))

    def _render_tree(self) -> Tree:
        """Build a hierarchical tree view of the pipeline steps.

        Returns:
            Rich Tree object with nested step information.
        """
        theme = self.theme
        tree = Tree(f"[{theme.header}]{self.title}[/{theme.header}]")

        # Group steps by parent
        root_steps: list[str] = []
        child_map: dict[str, list[str]] = defaultdict(list)

        for name in self.step_order:
            step = self.step_map[name]
            if step.parent:
                child_map[step.parent].append(name)
            else:
                root_steps.append(name)

        def _add_step(parent_node: Any, step_name: str) -> None:
            """Recursively add a step and its children to the tree."""
            step = self.step_map[step_name]
            icon = _STATUS_ICONS.get(step.status, "?")

            label_parts = [icon, step_name]
            if step.status == "running":
                elapsed = time.time() - step.start_time
                label_parts.append(f"[{theme.running}]{elapsed:.1f}s[/{theme.running}]")
            elif step.status == "done":
                label_parts.append(f"[{theme.done}]{step.duration:.1f}s[/{theme.done}]")
            elif step.status == "failed":
                label_parts.append(f"[{theme.failed}]{step.error}[/{theme.failed}]")

            node = parent_node.add(" ".join(label_parts))

            # Add children
            for child_name in child_map.get(step_name, []):
                _add_step(node, child_name)

        for step_name in root_steps:
            _add_step(tree, step_name)

        return tree

    def start(self) -> None:
        """Show the animated header panel and step list."""
        self._start_time = time.time()
        self._live = Live(self._render(), console=console, refresh_per_second=4)
        self._live.__enter__()

    def step_start(self, step_name: str) -> None:
        """Mark a step as in-progress.

        Args:
            step_name: Name of the step (must match one provided at init).
        """
        if step_name in self.step_map:
            self.step_map[step_name].status = "running"
            self.step_map[step_name].start_time = time.time()

            # Set ETA from historical data
            if self.eta_calculator:
                self.step_map[step_name].eta_seconds = self.eta_calculator.get_eta(step_name)

            self.callback.notify_step_start(step_name)
            self._refresh()

    def step_done(self, step_name: str, duration: float, detail: str = "") -> None:
        """Mark a step as completed.

        Args:
            step_name: Name of the step.
            duration: How long the step took in seconds.
            detail: Optional detail string (e.g. "42 words").
        """
        if step_name in self.step_map:
            self.step_map[step_name].status = "done"
            self.step_map[step_name].duration = duration
            self.step_map[step_name].detail = detail
            self.show_ffmpeg = False

            # Record in ETA calculator for future predictions
            if self.eta_calculator:
                self.eta_calculator.record_step(step_name, duration)

            self.callback.notify_step_done(step_name, duration)
            self._refresh()

    def step_failed(self, step_name: str, error: str) -> None:
        """Mark a step as failed.

        Args:
            step_name: Name of the step.
            error: Error message to display.
        """
        if step_name in self.step_map:
            self.step_map[step_name].status = "failed"
            self.step_map[step_name].error = error[:80]
            self.show_ffmpeg = False
            self.callback.notify_step_failed(step_name, error)
            self._refresh()

    def step_skipped(self, step_name: str, reason: str = "") -> None:
        """Mark a step as skipped.

        Args:
            step_name: Name of the step.
            reason: Optional reason for skipping.
        """
        if step_name in self.step_map:
            self.step_map[step_name].status = "skipped"
            self.step_map[step_name].detail = reason
            self._refresh()

    def update_step_progress(self, step_name: str, progress: float, detail: str = "") -> None:
        """Update sub-progress within a running step.

        Args:
            step_name: Name of the running step.
            progress: Progress value from 0.0 to 1.0.
            detail: Optional detail text to update.
        """
        if step_name in self.step_map and self.step_map[step_name].status == "running":
            self.step_map[step_name].progress = max(0.0, min(1.0, progress))
            if detail:
                self.step_map[step_name].detail = detail
            self._refresh()

    def update_ffmpeg(self, percent: float, speed: str, eta: str) -> None:
        """Update the FFmpeg sub-progress bar.

        Args:
            percent: Completion percentage (0-100).
            speed: Encoding speed string (e.g. '1.5x').
            eta: Estimated time remaining string.
        """
        self.show_ffmpeg = True
        self.ffmpeg_percent = percent
        self.ffmpeg_speed = speed
        self.ffmpeg_eta = eta
        self._refresh()

    def set_step_parent(self, step_name: str, parent_name: str) -> None:
        """Set a parent step for hierarchical display.

        Args:
            step_name: Name of the child step.
            parent_name: Name of the parent step.
        """
        if step_name in self.step_map and parent_name in self.step_map:
            self.step_map[step_name].parent = parent_name

    def set_step_parallel(self, step_name: str, can_parallel: bool = True) -> None:
        """Mark a step as eligible for parallel execution.

        Args:
            step_name: Name of the step.
            can_parallel: Whether the step can run in parallel.
        """
        if step_name in self.step_map:
            self.step_map[step_name].can_parallel = can_parallel

    def get_parallel_groups(self) -> list[list[str]]:
        """Identify groups of steps that can run in parallel.

        Returns:
            List of step groups. Each group is a list of step names
            that can run concurrently. Sequential steps are groups of one.
        """
        groups: list[list[str]] = []
        current_group: list[str] = []

        for name in self.step_order:
            step = self.step_map[name]
            if step.can_parallel:
                current_group.append(name)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([name])

        if current_group:
            groups.append(current_group)

        return groups

    def export_progress(self) -> dict[str, Any]:
        """Export current progress state as a dictionary.

        Returns:
            Dictionary with full progress state for external consumption.
        """
        return self.exporter.export_to_dict()

    def export_progress_json(self) -> str:
        """Export current progress state as a JSON string.

        Returns:
            Formatted JSON string of the progress state.
        """
        return self.exporter.export_to_string()

    def export_progress_to_file(self, path: Path) -> None:
        """Export current progress state to a JSON file.

        Args:
            path: File path to write the progress export.
        """
        self.exporter.export_to_file(path)

    def finish(self, summary: dict) -> None:
        """Show a final Rich Panel with all stats and stop the live display.

        Args:
            summary: Dict with keys like 'outputs', 'total_time', 'file_sizes', etc.
        """
        if self._live:
            self._live.__exit__(None, None, None)
            self._live = None

        # Build final summary panel
        panel_content = Table(show_header=False, box=None, padding=(0, 1))
        panel_content.add_column("key", style="bold cyan")
        panel_content.add_column("value")

        for key, value in summary.items():
            panel_content.add_row(str(key), str(value))

        total_time = time.time() - self._start_time if self._start_time else 0
        panel_content.add_row("Total Time", f"{total_time:.1f}s")

        # Show step breakdown
        panel_content.add_row("", "")
        panel_content.add_row("[bold]Step Breakdown[/bold]", "")
        for name in self.step_order:
            step = self.step_map[name]
            status_icon = _STATUS_ICONS.get(step.status, "?")
            duration_str = f"{step.duration:.1f}s" if step.duration > 0 else "-"
            panel_content.add_row(
                f"  {status_icon} {name}",
                f"{duration_str} {step.detail}".strip(),
            )

        console.print()
        console.print(
            Panel(
                panel_content,
                title="[bold green]Pipeline Complete[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

        # Notify callbacks with final progress
        self.callback.notify_progress_update(self.export_progress())

    def _refresh(self) -> None:
        """Refresh the live display if active."""
        if self._live:
            self._live.update(self._render())
