"""
utils/logger.py — Structured rotating file logger with Rich console output.

Provides get_logger(name) factory that returns a configured Python logger
with both a rotating file handler and a colour-coded Rich console handler.
Supports JSON structured logging, per-module log level overrides via
environment variables, execution timing decorator, context injection,
and audit logging. Thread-safe and safe for repeated calls.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
import threading
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Generator

from rich.logging import RichHandler


# ── Constants ──────────────────────────────────────────────
_LOG_DIR: Path = Path(__file__).resolve().parent.parent / "output" / "logs"
_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT: int = 7
_LOG_FORMAT: str = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# Thread-safe guard for logger creation
_logger_lock = threading.Lock()
_created_loggers: set[str] = set()

# Environment variable names
_ENV_LOG_LEVEL = "LOG_LEVEL"
_ENV_LOG_FORMAT = "LOG_FORMAT"  # 'text' or 'json'
_ENV_LOG_LEVEL_PREFIX = "LOG_LEVEL_"  # e.g. LOG_LEVEL_ffmpeg_utils=DEBUG


def _resolve_log_level(name: str, default: int = logging.DEBUG) -> int:
    """Resolve the log level for a given logger name.

    Checks in order:
    1. LOG_LEVEL_{name} environment variable (per-module override)
    2. LOG_LEVEL environment variable (global override)
    3. The default level

    Args:
        name: Logger name to resolve level for.
        default: Default log level if no override found.

    Returns:
        Resolved log level as an integer.
    """
    # Check per-module override first
    module_env = f"{_ENV_LOG_LEVEL_PREFIX}{name}"
    module_level_str = os.environ.get(module_env, "")
    if module_level_str:
        level = _parse_level_string(module_level_str)
        if level is not None:
            return level

    # Check global override
    global_level_str = os.environ.get(_ENV_LOG_LEVEL, "")
    if global_level_str:
        level = _parse_level_string(global_level_str)
        if level is not None:
            return level

    return default


def _parse_level_string(level_str: str) -> int | None:
    """Parse a log level string into an integer.

    Args:
        level_str: Log level name (e.g. 'DEBUG', 'INFO') or numeric string.

    Returns:
        Log level integer, or None if the string is invalid.
    """
    level_str = level_str.strip().upper()

    # Try named levels
    named_levels = {
        "NOTSET": logging.NOTSET,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,
    }
    if level_str in named_levels:
        return named_levels[level_str]

    # Try numeric level
    try:
        return int(level_str)
    except ValueError:
        return None


def _is_json_format() -> bool:
    """Check if JSON structured logging is requested via environment variable.

    Returns:
        True if LOG_FORMAT=json is set in the environment.
    """
    return os.environ.get(_ENV_LOG_FORMAT, "text").strip().lower() == "json"


# ══════════════════════════════════════════════════════════
#  JSON Formatter
# ══════════════════════════════════════════════════════════

class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging.

    Each log record is serialised as a single-line JSON object with
    standard fields (timestamp, level, logger, message) plus any
    extra context fields attached to the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            Single-line JSON string.
        """
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }

        # Include any extra fields from the record
        standard_attrs = {
            "name", "msg", "args", "created", "relativeCreated",
            "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "pathname", "filename", "module", "levelno", "levelname",
            "thread", "threadName", "process", "processName", "message",
            "msecs", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                try:
                    json.dumps(value)  # Test serialisability
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        try:
            return json.dumps(log_entry, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return json.dumps({
                "timestamp": log_entry["timestamp"],
                "level": log_entry["level"],
                "logger": log_entry["logger"],
                "message": record.getMessage(),
            }, ensure_ascii=False)


# ══════════════════════════════════════════════════════════
#  Context Injection Filter
# ══════════════════════════════════════════════════════════

_context_data: threading.local = threading.local()


def _get_context() -> dict[str, Any]:
    """Get the current thread-local logging context.

    Returns:
        Dictionary of context key-value pairs.
    """
    return getattr(_context_data, "data", {})


def _set_context(data: dict[str, Any]) -> None:
    """Set the thread-local logging context.

    Args:
        data: Dictionary of context key-value pairs.
    """
    _context_data.data = data


class ContextFilter(logging.Filter):
    """Inject thread-local context data into log records.

    Adds all key-value pairs from the current logging context to each
    log record, making them available to formatters and handlers.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context fields to the log record.

        Args:
            record: The log record to enrich.

        Returns:
            Always True (never filters out records).
        """
        context = _get_context()
        for key, value in context.items():
            setattr(record, key, value)
        return True


# ══════════════════════════════════════════════════════════
#  Logger Factory
# ══════════════════════════════════════════════════════════

def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Create or retrieve a logger with rotating file + Rich console handlers.

    Thread-safe. Prevents duplicate handlers on repeated calls. The file handler
    writes to output/logs/{name}_{YYYYMMDD}.log with rotation at 10 MB and
    7 backups retained. The console handler uses Rich for colour-coded output.

    Supports:
    - JSON structured logging (set LOG_FORMAT=json env var)
    - Per-module log level override (set LOG_LEVEL_{name} env var)
    - Global log level override (set LOG_LEVEL env var)
    - Context injection via log_context() context manager

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Minimum log level. If None, resolved from env vars or DEBUG.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Fast path: already configured
    if name in _created_loggers:
        return logger

    with _logger_lock:
        # Double-check under lock
        if name in _created_loggers:
            return logger

        # Resolve log level
        resolved_level = level if level is not None else _resolve_log_level(name)
        logger.setLevel(resolved_level)
        logger.propagate = False

        # Avoid duplicate handlers
        if logger.handlers:
            _created_loggers.add(name)
            return logger

        # ── Ensure log directory exists ──────────────────
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # Best effort

        # ── Context filter (applies to all handlers) ─────
        context_filter = ContextFilter()
        logger.addFilter(context_filter)

        # ── Choose formatter based on env var ────────────
        use_json = _is_json_format()

        if use_json:
            file_formatter: logging.Formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

        # ── Rotating file handler (by size) ──────────────
        date_stamp: str = datetime.now().strftime("%Y%m%d")
        log_file: Path = _LOG_DIR / f"{name}_{date_stamp}.log"
        try:
            file_handler = RotatingFileHandler(
                filename=str(log_file),
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(context_filter)
            logger.addHandler(file_handler)
        except OSError as exc:
            import warnings
            warnings.warn(f"Could not create log file handler for {name}: {exc}")

        # ── Timed rotating file handler (daily rotation) ──
        timed_log_file: Path = _LOG_DIR / f"{name}_daily.log"
        try:
            timed_handler = TimedRotatingFileHandler(
                filename=str(timed_log_file),
                when="midnight",
                interval=1,
                backupCount=30,
                encoding="utf-8",
            )
            timed_handler.setLevel(logging.DEBUG)
            timed_handler.setFormatter(file_formatter)
            timed_handler.addFilter(context_filter)
            logger.addHandler(timed_handler)
        except OSError as exc:
            import warnings
            warnings.warn(f"Could not create timed log file handler for {name}: {exc}")

        # ── Rich console handler ─────────────────────────
        console_handler = RichHandler(
            level=logging.INFO,
            rich_tracebacks=True,
            show_path=True,
            markup=True,
            show_time=True,
            show_level=True,
        )
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(context_filter)
        logger.addHandler(console_handler)

        _created_loggers.add(name)

    return logger


# ══════════════════════════════════════════════════════════
#  Log Context Manager
# ══════════════════════════════════════════════════════════

@contextmanager
def log_context(**kwargs: Any) -> Generator[None, None, None]:
    """Context manager that adds key-value pairs to all log messages in the block.

    Thread-local context is automatically merged and restored when the
    context exits. Nested contexts are supported — inner contexts extend
    outer contexts.

    Args:
        **kwargs: Key-value pairs to inject into log records.

    Example:
        with log_context(job_id="abc123", stage="transcription"):
            logger.info("Starting transcription")  # Will include job_id and stage
    """
    previous_context = _get_context().copy()
    current_context = previous_context.copy()
    current_context.update(kwargs)
    _set_context(current_context)
    try:
        yield
    finally:
        _set_context(previous_context)


# ══════════════════════════════════════════════════════════
#  Execution Timing Decorator
# ══════════════════════════════════════════════════════════

def log_execution_time(
    logger_instance: logging.Logger | None = None,
    level: int = logging.INFO,
    prefix: str = "",
) -> Callable:
    """Decorator that logs the execution time of a function.

    Measures wall-clock time and logs it at the specified level. If no
    logger is provided, uses the 'timing' logger.

    Args:
        logger_instance: Logger to use. If None, creates a 'timing' logger.
        level: Log level for the timing message (default INFO).
        prefix: Optional prefix for the log message.

    Returns:
        Decorator function.

    Example:
        @log_execution_time(prefix="Pipeline")
        def run_pipeline():
            ...
    """
    _logger = logger_instance or get_logger("timing")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = getattr(func, "__name__", str(func))
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                msg = f"{prefix}{func_name} completed in {elapsed:.3f}s" if prefix else f"{func_name} completed in {elapsed:.3f}s"
                _logger.log(level, msg, extra={"duration_seconds": round(elapsed, 3), "function": func_name})
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                msg = f"{prefix}{func_name} failed after {elapsed:.3f}s: {exc}" if prefix else f"{func_name} failed after {elapsed:.3f}s: {exc}"
                _logger.error(msg, extra={"duration_seconds": round(elapsed, 3), "function": func_name, "error": str(exc)})
                raise

        return wrapper

    return decorator


# ══════════════════════════════════════════════════════════
#  Audit Logger
# ══════════════════════════════════════════════════════════

_audit_logger_instance: logging.Logger | None = None
_audit_lock = threading.Lock()


def get_audit_logger() -> logging.Logger:
    """Get or create a dedicated audit trail logger.

    The audit logger writes to a separate file (audit.log) with JSON
    formatting for easy parsing and compliance. It always logs at
    INFO level and above, and uses a TimedRotatingFileHandler with
    90-day retention.

    Returns:
        Configured audit Logger instance.
    """
    global _audit_logger_instance

    if _audit_logger_instance is not None:
        return _audit_logger_instance

    with _audit_lock:
        if _audit_logger_instance is not None:
            return _audit_logger_instance

        audit_logger = logging.getLogger("audit")

        if audit_logger.handlers:
            _audit_logger_instance = audit_logger
            return audit_logger

        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False

        # Ensure log directory exists
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        # JSON-formatted audit file with 90-day retention
        audit_file = _LOG_DIR / "audit.log"
        try:
            handler = TimedRotatingFileHandler(
                filename=str(audit_file),
                when="midnight",
                interval=1,
                backupCount=90,
                encoding="utf-8",
            )
            handler.setLevel(logging.INFO)
            handler.setFormatter(JSONFormatter())

            # Add context filter for audit records
            context_filter = ContextFilter()
            handler.addFilter(context_filter)

            audit_logger.addHandler(handler)
        except OSError as exc:
            import warnings
            warnings.warn(f"Could not create audit log handler: {exc}")

        _audit_logger_instance = audit_logger

    return _audit_logger_instance


# ══════════════════════════════════════════════════════════
#  Batch Log Aggregation
# ══════════════════════════════════════════════════════════

class BatchLogAggregator:
    """Aggregate log messages for batch operations.

    Collects log messages during a batch operation and provides a
    summary at the end. Useful for reducing log noise when processing
    many items — instead of logging each item individually, the
    aggregator collects results and emits a single summary.

    Example:
        agg = BatchLogAggregator("video_export")
        for video in videos:
            with agg:
                export_video(video)
        agg.summarize()  # Logs summary: "video_export: 45 succeeded, 5 failed"
    """

    def __init__(
        self,
        operation_name: str,
        logger_instance: logging.Logger | None = None,
    ) -> None:
        """Initialise the batch log aggregator.

        Args:
            operation_name: Name of the batch operation for log messages.
            logger_instance: Logger to use. If None, creates one from operation_name.
        """
        self.operation_name = operation_name
        self._logger = logger_instance or get_logger(f"batch.{operation_name}")
        self._successes: int = 0
        self._failures: int = 0
        self._errors: list[str] = []
        self._start_time: float = 0.0
        self._item_start: float = 0.0

    def __enter__(self) -> BatchLogAggregator:
        """Start timing a single batch item."""
        self._item_start = time.perf_counter()
        if self._start_time == 0.0:
            self._start_time = self._item_start
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Record the result of a single batch item."""
        if exc_type is None:
            self._successes += 1
        else:
            self._failures += 1
            error_msg = str(exc_val) if exc_val else "Unknown error"
            self._errors.append(error_msg[:200])

    def record_success(self, detail: str = "") -> None:
        """Manually record a successful item.

        Args:
            detail: Optional detail string.
        """
        self._successes += 1

    def record_failure(self, error: str = "") -> None:
        """Manually record a failed item.

        Args:
            error: Error description.
        """
        self._failures += 1
        if error:
            self._errors.append(error[:200])

    def summarize(self) -> dict[str, Any]:
        """Log and return a summary of the batch operation.

        Returns:
            Dictionary with 'operation', 'successes', 'failures',
            'total', 'duration_seconds', and 'errors'.
        """
        elapsed = time.perf_counter() - self._start_time if self._start_time > 0 else 0.0
        total = self._successes + self._failures

        summary = {
            "operation": self.operation_name,
            "successes": self._successes,
            "failures": self._failures,
            "total": total,
            "duration_seconds": round(elapsed, 2),
            "errors": self._errors[:10],  # Cap error list
        }

        if self._failures == 0:
            self._logger.info(
                "%s: %d succeeded in %.1fs",
                self.operation_name, self._successes, elapsed,
                extra=summary,
            )
        else:
            self._logger.warning(
                "%s: %d succeeded, %d failed in %.1fs",
                self.operation_name, self._successes, self._failures, elapsed,
                extra=summary,
            )

        return summary

    @property
    def success_count(self) -> int:
        """Return the number of successful items."""
        return self._successes

    @property
    def failure_count(self) -> int:
        """Return the number of failed items."""
        return self._failures
