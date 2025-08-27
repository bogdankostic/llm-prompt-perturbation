import time
import random
from typing import Callable, Any
from functools import wraps
from google.api_core import exceptions


class RateLimiter:
    """
    A utility class for handling rate limiting and retries for API calls.
    """
    
    def __init__(
        self,
        calls_per_second: float = 2.0,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """
        Initialize the rate limiter.
        
        :param calls_per_second: Maximum number of API calls per second
        :param max_retries: Maximum number of retries for failed calls
        :param base_delay: Base delay for exponential backoff in seconds
        :param max_delay: Maximum delay between retries in seconds
        """
        self.calls_per_second = calls_per_second
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.last_call_time = 0.0
    
    def wait_if_needed(self):
        """Wait if necessary to respect the rate limit."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        min_interval = 1.0 / self.calls_per_second
        
        if time_since_last_call < min_interval:
            sleep_time = min_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic and exponential backoff.
        
        :param func: The function to execute
        :param args: Positional arguments for the function
        :param kwargs: Keyword arguments for the function
        :return: The result of the function call
        """
        for attempt in range(self.max_retries):
            try:
                self.wait_if_needed()
                return func(*args, **kwargs)
            except exceptions.ResourceExhausted as e:
                if attempt == self.max_retries - 1:
                    raise e
                
                # Exponential backoff with jitter
                delay = min(
                    self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                    self.max_delay
                )
                print(f"Rate limited, retrying in {delay:.2f} seconds... (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                
                # For other errors, retry with shorter delay
                delay = min(
                    self.base_delay * (1.5 ** attempt) + random.uniform(0, 0.5),
                    self.max_delay
                )
                print(f"API error, retrying in {delay:.2f} seconds... (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)


def rate_limited(calls_per_second: float = 2.0):
    """
    Decorator to rate limit function calls.
    
    :param calls_per_second: Maximum number of calls per second
    """
    def decorator(func: Callable) -> Callable:
        limiter = RateLimiter(calls_per_second=calls_per_second)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait_if_needed()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 5, base_delay: float = 1.0):
    """
    Decorator to retry function calls on failure with exponential backoff.
    
    :param max_retries: Maximum number of retries
    :param base_delay: Base delay for exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        limiter = RateLimiter(max_retries=max_retries, base_delay=base_delay)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return limiter.retry_with_backoff(func, *args, **kwargs)
        
        return wrapper
    return decorator
