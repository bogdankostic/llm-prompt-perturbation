import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from haystack import component, default_to_dict
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage


@component
class CachedOpenAIChatGenerator(OpenAIChatGenerator):
    """
    A component that extends OpenAIChatGenerator with caching functionality.
    
    This component caches the responses from the OpenAI API and returns cached responses
    when the same messages are provided again. This helps reduce API calls and costs.
    The cache is organized by model name to prevent mixing responses from different models.
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache",
        *args,
        **kwargs
    ) -> None:
        """
        Initialize the CachedOpenAIChatGenerator component.

        :param cache_dir: Directory to store the cache files
        :param *args: Arguments to pass to OpenAIChatGenerator
        :param **kwargs: Keyword arguments to pass to OpenAIChatGenerator
        """
        super().__init__(*args, **kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_model_cache_dir(self) -> Path:
        """
        Get the cache directory for the current model.

        :return: Path to the model-specific cache directory
        """
        # Create a safe directory name from the model name
        model_dir = self.model.replace("/", "_")
        model_cache_dir = self.cache_dir / model_dir
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        return model_cache_dir
    
    def _get_cache_key(self, messages: List[ChatMessage]) -> str:
        """
        Generate a cache key from the messages.

        :param messages: List of chat messages
        :return: A string key for caching
        """
        # Convert messages to a string representation
        messages_str = json.dumps([msg.to_dict() for msg in messages])
        # Use a hash of the messages as the cache key
        return str(hash(messages_str))
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response if it exists.

        :param cache_key: The cache key to look up
        :return: The cached response or None if not found
        """
        model_cache_dir = self._get_model_cache_dir()
        cache_file = model_cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)
        return None
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]) -> None:
        """
        Cache a response.

        :param cache_key: The cache key to store under
        :param response: The response to cache
        """
        model_cache_dir = self._get_model_cache_dir()
        cache_file = model_cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(response, f)
    
    def run(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """
        Run the chat generator with caching.

        :param messages: List of chat messages
        :return: The generated response
        """
        cache_key = self._get_cache_key(messages)
        cached_response = self._get_cached_response(cache_key)
        
        if cached_response is not None:
            return cached_response
        
        # If not in cache, generate response using parent class
        response = super().run(messages)
        
        # Cache the response
        self._cache_response(cache_key, response)
        
        return response
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the component's configuration to a dictionary.

        :return: A dictionary containing the component's configuration parameters.
        """
        return default_to_dict(
            self,
            cache_dir=str(self.cache_dir),
            **super().to_dict()
        )
