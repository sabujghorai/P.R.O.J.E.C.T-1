
"""
RETRY UTILITY

Calls a function and, if it raises an exception, retries
a few times with exponential backoff.

Used for Groq and Tavily API calls so temporary rate limits
or network issues don't immediately fail the request.

Example:
    response = with_retry(
        lambda: groq_client.chat.completions.create(
            model="your-model",
            messages=[{"role": "user", "content": "Hello"}]
        ),
        max_retries=3,
        initial_delay=1.0,
    )
"""

import logging
import time
from typing import TypeVar, Callable

logger = logging.getLogger("J.A.R.V.I.S")

# Type variable: with_retry returns whatever the callable returns.
T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
) -> T:
    """
    Execute fn(). If it raises an exception, wait initial_delay
    seconds and try again. The delay doubles after each retry.
    Args:
        fn: Function to execute.
        max_retries: Total attempts, including the first attempt.
        initial_delay: Initial wait time in seconds.
    Returns:
        Whatever the function returns.
    Raises:
        ValueError: If max_retries is less than 1 or
        initial_delay is negative.
        Exception: The last exception if all attempts fail.
    """

    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    if initial_delay < 0:
        raise ValueError("initial_delay cannot be negative")

    delay = initial_delay

    for attempt in range(max_retries):
        try:
            # Execute the function and return its result.
            return fn()

        except Exception as e:
            # Check whether this was the final attempt.
            if attempt == max_retries - 1:
                logger.error(
                    "All %s attempts failed. Final error: %s",
                    max_retries,
                    e,
                )
                raise

            # Log the failed attempt and retry delay.
            logger.warning(
                "Attempt %s/%s failed (%s). "
                "Retrying in %.1fs: %s",
                attempt + 1,
                max_retries,
                getattr(fn, "__name__", "call"),
                delay,
                e,
            )

            # Wait before the next attempt.
            time.sleep(delay)

            # Exponential backoff: 1s, 2s, 4s, 8s...
            delay *= 2