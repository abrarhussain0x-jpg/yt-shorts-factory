"""
utils/retry.py — Exponential backoff retry, circuit breaker, and rate limiter.

Provides:
- retry_on_failure: Decorator with exponential backoff and jitter
- retry_with_fallback: Decorator that tries fallback functions on failure
- retry_if_result: Decorator that retries based on return value, not exception
- CircuitBreaker: Prevents repeated calls to failing services
- RateLimiter: Controls call frequency to avoid overwhelming services
All decorators include detailed logging and metrics collection.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Callable, Tuple, Type

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
    retry_if_exception_type,
    after_log,
)

from utils.logger import get_logger

logger = get_logger("retry")


# ══════════════════════════════════════════════════════════
#  retry_on_failure — Enhanced with metrics
# ══════════════════════════════════════════════════════════

# Global metrics for retry operations
_retry_metrics: dict[str, dict[str, int]] = {}
_metrics_lock = threading.Lock()


def _record_retry_metric(func_name: str, success: bool) -> None:
    """Record retry metrics for monitoring.

    Args:
        func_name: Name of the function that was retried.
        success: Whether the function eventually succeeded.
    """
    with _metrics_lock:
        if func_name not in _retry_metrics:
            _retry_metrics[func_name] = {"attempts": 0, "successes": 0, "failures": 0}
        _retry_metrics[func_name]["attempts"] += 1
        if success:
            _retry_metrics[func_name]["successes"] += 1
        else:
            _retry_metrics[func_name]["failures"] += 1


def get_retry_metrics() -> dict[str, dict[str, int]]:
    """Get a snapshot of current retry metrics.

    Returns:
        Dictionary mapping function names to their retry metrics
        (attempts, successes, failures).
    """
    with _metrics_lock:
        return dict(_retry_metrics)


def reset_retry_metrics() -> None:
    """Reset all retry metrics."""
    with _metrics_lock:
        _retry_metrics.clear()


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 5.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    max_delay: float = 120.0,
) -> Callable:
    """Decorator that retries a function with exponential backoff and jitter.

    Uses tenacity's wait_exponential_jitter for intelligent backoff.
    Logs each retry attempt with full context and records metrics.

    Args:
        max_attempts: Maximum number of attempts before giving up.
        delay: Base delay in seconds between retries. Actual delay follows
               exponential backoff: delay * 2^attempt + random(0, 1).
        exceptions: Tuple of exception types that should trigger a retry.
        max_delay: Maximum delay in seconds between retries.

    Returns:
        Decorated function that will retry on specified exceptions.

    Raises:
        The last exception encountered after all attempts are exhausted.

    Example:
        @retry_on_failure(max_attempts=3, delay=2.0, exceptions=(subprocess.CalledProcessError,))
        def run_ffmpeg(cmd):
            ...
    """

    def decorator(func: Callable) -> Callable:
        retry_decorator = retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=delay, exp_base=2, jitter=1, max=max_delay),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.DEBUG),
            reraise=True,
        )

        @retry_decorator
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = getattr(func, "__name__", str(func))
            try:
                result = func(*args, **kwargs)
                _record_retry_metric(func_name, success=True)
                return result
            except Exception:
                _record_retry_metric(func_name, success=False)
                raise

        return wrapper

    return decorator


# ══════════════════════════════════════════════════════════
#  retry_with_fallback
# ══════════════════════════════════════════════════════════

def retry_with_fallback(
    *fallback_funcs: Callable,
    max_attempts: int = 2,
    delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """Decorator that retries a function, then tries fallback functions on failure.

    First attempts the primary function up to max_attempts times. If all
    attempts fail, each fallback function is tried in order. The first
    successful result is returned.

    Args:
        *fallback_funcs: Fallback functions to try in order after the primary fails.
        max_attempts: Maximum attempts for the primary function before trying fallbacks.
        delay: Delay between retry attempts in seconds.
        exceptions: Exception types that trigger retry or fallback.

    Returns:
        Decorated function with retry and fallback logic.

    Raises:
        The last exception if all functions (primary + fallbacks) fail.

    Example:
        def primary_download(url):
            ...

        def fallback_download(url):
            ...

        @retry_with_fallback(fallback_download, max_attempts=2)
        def download(url):
            return primary_download(url)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = getattr(func, "__name__", str(func))
            last_exception: Exception | None = None

            # Try primary function
            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    _record_retry_metric(func_name, success=True)
                    return result
                except exceptions as exc:
                    last_exception = exc
                    logger.warning(
                        "%s attempt %d/%d failed: %s",
                        func_name, attempt, max_attempts, exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(delay * (2 ** (attempt - 1)))

            # Try fallback functions
            for idx, fallback in enumerate(fallback_funcs, start=1):
                fallback_name = getattr(fallback, "__name__", f"fallback_{idx}")
                try:
                    logger.info("Trying %s as fallback %d for %s", fallback_name, idx, func_name)
                    result = fallback(*args, **kwargs)
                    _record_retry_metric(f"{func_name}_fallback_{idx}", success=True)
                    logger.info("Fallback %d (%s) succeeded for %s", idx, fallback_name, func_name)
                    return result
                except exceptions as exc:
                    last_exception = exc
                    logger.warning(
                        "Fallback %d (%s) failed for %s: %s",
                        idx, fallback_name, func_name, exc,
                    )
                    _record_retry_metric(f"{func_name}_fallback_{idx}", success=False)

            # All attempts and fallbacks exhausted
            _record_retry_metric(func_name, success=False)
            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"All functions failed for {func_name} with no captured exception")

        return wrapper

    return decorator


# ══════════════════════════════════════════════════════════
#  retry_if_result
# ══════════════════════════════════════════════════════════

def retry_if_result(
    condition: Callable[[Any], bool],
    max_attempts: int = 3,
    delay: float = 1.0,
    max_delay: float = 30.0,
) -> Callable:
    """Decorator that retries based on the return value, not exceptions.

    Retries the function if the condition applied to the return value
    returns True. Useful for cases where the function "succeeds" (no
    exception) but returns an unacceptable result.

    Args:
        condition: Callable that takes the return value and returns True
                   if a retry is needed (result is unacceptable).
        max_attempts: Maximum number of attempts.
        delay: Base delay in seconds between retries.
        max_delay: Maximum delay in seconds between retries.

    Returns:
        Decorated function with result-based retry logic.

    Raises:
        ValueError: If all attempts produce unacceptable results.

    Example:
        @retry_if_result(lambda r: r is None, max_attempts=5)
        def find_best_clip():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = getattr(func, "__name__", str(func))
            last_result: Any = None

            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    last_result = result

                    if not condition(result):
                        _record_retry_metric(func_name, success=True)
                        return result

                    logger.debug(
                        "%s returned unacceptable result (attempt %d/%d), retrying",
                        func_name, attempt, max_attempts,
                    )
                except Exception as exc:
                    logger.warning(
                        "%s raised exception on attempt %d/%d: %s",
                        func_name, attempt, max_attempts, exc,
                    )
                    _record_retry_metric(func_name, success=False)
                    raise

                if attempt < max_attempts:
                    sleep_time = min(delay * (2 ** (attempt - 1)), max_delay)
                    time.sleep(sleep_time)

            _record_retry_metric(func_name, success=False)
            logger.warning(
                "%s returned unacceptable result after %d attempts",
                func_name, max_attempts,
            )
            raise ValueError(
                f"{func_name} returned unacceptable result after {max_attempts} attempts. "
                f"Last result: {last_result!r}"
            )

        return wrapper

    return decorator


# ══════════════════════════════════════════════════════════
#  CircuitBreaker
# ══════════════════════════════════════════════════════════

class CircuitBreaker:
    """Prevents repeated calls to a failing service.

    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, calls pass through. Failures are counted.
    - OPEN: Too many failures, calls are rejected immediately.
    - HALF_OPEN: Testing if the service has recovered. One call is allowed.

    Args:
        failure_threshold: Number of failures before opening the circuit.
        recovery_timeout: Seconds to wait before transitioning to HALF_OPEN.
        expected_exceptions: Exception types that count as failures.

    Example:
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

        @breaker
        def call_api():
            ...
    """

    # Circuit states
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ) -> None:
        """Initialise the circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening (default 5).
            recovery_timeout: Seconds in OPEN state before trying HALF_OPEN (default 30).
            expected_exceptions: Exception types that count as failures.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._last_failure_exception: Exception | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Return the current state of the circuit breaker."""
        with self._lock:
            if self._state == self.OPEN:
                # Check if recovery timeout has elapsed
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
            return self._state

    @property
    def failure_count(self) -> int:
        """Return the current consecutive failure count."""
        return self._failure_count

    @property
    def success_count(self) -> int:
        """Return the total success count since last reset."""
        return self._success_count

    def __call__(self, func: Callable) -> Callable:
        """Use the circuit breaker as a decorator.

        Args:
            func: Function to wrap with circuit breaker logic.

        Returns:
            Wrapped function.
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a function through the circuit breaker.

        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The function's return value.

        Raises:
            RuntimeError: If the circuit is OPEN and calls are rejected.
        """
        current_state = self.state
        func_name = getattr(func, "__name__", str(func))

        if current_state == self.OPEN:
            raise RuntimeError(
                f"Circuit breaker is OPEN for {func_name}. "
                f"Last failure: {self._last_failure_exception}. "
                f"Retry after {self.recovery_timeout}s."
            )

        try:
            result = func(*args, **kwargs)
            self._on_success(func_name)
            return result
        except self.expected_exceptions as exc:
            self._on_failure(func_name, exc)
            raise

    def _on_success(self, func_name: str) -> None:
        """Handle a successful call.

        Args:
            func_name: Name of the function that succeeded.
        """
        with self._lock:
            self._failure_count = 0
            self._success_count += 1

            if self._state == self.HALF_OPEN:
                self._state = self.CLOSED
                logger.info("Circuit breaker recovered, transitioning to CLOSED for %s", func_name)

    def _on_failure(self, func_name: str, exc: Exception) -> None:
        """Handle a failed call.

        Args:
            func_name: Name of the function that failed.
            exc: The exception that was raised.
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            self._last_failure_exception = exc

            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
                logger.warning(
                    "Circuit breaker HALF_OPEN -> OPEN for %s (failure during test call)",
                    func_name,
                )
            elif self._failure_count >= self.failure_threshold:
                self._state = self.OPEN
                logger.warning(
                    "Circuit breaker CLOSED -> OPEN for %s (%d consecutive failures)",
                    func_name, self._failure_count,
                )

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = 0.0
            self._last_failure_exception = None
            logger.info("Circuit breaker manually reset to CLOSED")

    def get_stats(self) -> dict[str, Any]:
        """Get current circuit breaker statistics.

        Returns:
            Dictionary with state, failure_count, success_count,
            last_failure_time, and last_failure_exception.
        """
        return {
            "state": self.state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "last_failure_exception": str(self._last_failure_exception) if self._last_failure_exception else None,
        }


# ══════════════════════════════════════════════════════════
#  RateLimiter
# ══════════════════════════════════════════════════════════

class RateLimiter:
    """Controls call frequency to avoid overwhelming services.

    Uses a token bucket algorithm to limit the rate of function calls.
    Thread-safe and suitable for use in concurrent environments.

    Args:
        calls_per_second: Maximum number of calls allowed per second.
        burst: Maximum burst size (default equals calls_per_second).

    Example:
        limiter = RateLimiter(calls_per_second=2.0)

        @limiter
        def call_api():
            ...
    """

    def __init__(
        self,
        calls_per_second: float = 1.0,
        burst: int | None = None,
    ) -> None:
        """Initialise the rate limiter.

        Args:
            calls_per_second: Maximum calls per second (default 1.0).
            burst: Maximum burst size. Defaults to int(calls_per_second).

        Raises:
            ValueError: If calls_per_second is not positive.
        """
        if calls_per_second <= 0:
            raise ValueError(f"calls_per_second must be positive, got {calls_per_second}")

        self.calls_per_second = calls_per_second
        self.burst = burst if burst is not None else max(1, int(calls_per_second))
        self._tokens: float = float(self.burst)
        self._last_refill: float = time.monotonic()
        self._lock = threading.Lock()
        self._total_calls = 0
        self._total_waited = 0.0

    def __call__(self, func: Callable) -> Callable:
        """Use the rate limiter as a decorator.

        Args:
            func: Function to rate-limit.

        Returns:
            Wrapped function with rate limiting.
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a function through the rate limiter.

        Blocks if necessary to respect the rate limit.

        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The function's return value.
        """
        self._acquire()
        return func(*args, **kwargs)

    def _acquire(self) -> None:
        """Acquire a token, blocking if necessary.

        Refills tokens based on elapsed time, then waits if no tokens
        are available.
        """
        with self._lock:
            self._refill()

            if self._tokens < 1.0:
                # Calculate wait time
                wait_time = (1.0 - self._tokens) / self.calls_per_second
                self._total_waited += wait_time

            while self._tokens < 1.0:
                # Release lock while waiting
                self._lock.release()
                try:
                    time.sleep(0.01)
                finally:
                    self._lock.acquire()
                self._refill()

            self._tokens -= 1.0
            self._total_calls += 1

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now

        new_tokens = elapsed * self.calls_per_second
        self._tokens = min(self._tokens + new_tokens, float(self.burst))

    @property
    def available_tokens(self) -> float:
        """Return the current number of available tokens."""
        with self._lock:
            self._refill()
            return self._tokens

    @property
    def total_calls(self) -> int:
        """Return the total number of calls made through this limiter."""
        return self._total_calls

    @property
    def total_waited(self) -> float:
        """Return the total time spent waiting in seconds."""
        return self._total_waited

    def get_stats(self) -> dict[str, Any]:
        """Get current rate limiter statistics.

        Returns:
            Dictionary with calls_per_second, burst, available_tokens,
            total_calls, and total_waited.
        """
        return {
            "calls_per_second": self.calls_per_second,
            "burst": self.burst,
            "available_tokens": round(self.available_tokens, 2),
            "total_calls": self._total_calls,
            "total_waited": round(self._total_waited, 3),
        }
