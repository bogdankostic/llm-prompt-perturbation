import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import hashlib

from haystack import component, default_to_dict
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import Tool, Toolset


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
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        super(CachedOpenAIChatGenerator, self).__init__(*args, **kwargs)
    
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
    
    def _get_cache_key(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a cache key from the messages.

        :param messages: List of chat messages
        :return: A string key for caching
        """
        # Convert messages to a list of dictionaries
        messages_dict = [msg.to_dict() for msg in messages]
        if generation_kwargs is not None:
            messages_dict.append({"generation_kwargs": generation_kwargs})
        # Convert to a string representation
        messages_str = json.dumps(messages_dict, sort_keys=True)
        # Use SHA-256 hash of the messages as the cache key
        return hashlib.sha256(messages_str.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response if it exists.

        :param cache_key: The cache key to look up
        :return: The cached response or None if not found
        """
        model_cache_dir = self._get_model_cache_dir()
        cache_file = model_cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    content = f.read()
                    if not content:  # Handle empty file
                        return None
                    response = json.loads(content)
                    # Convert dictionaries back to ChatMessage objects
                    if 'replies' in response:
                        response['replies'] = [ChatMessage.from_dict(msg) for msg in response['replies']]
                    return response
            except json.JSONDecodeError:
                # If the file contains invalid JSON, delete it and return None
                cache_file.unlink()
                return None
        return None
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]) -> None:
        """
        Cache a response.

        :param cache_key: The cache key to store under
        :param response: The response to cache
        """
        # Convert ChatMessage objects in replies to dictionaries
        if 'replies' in response:
            response['replies'] = [msg.to_dict() for msg in response['replies']]

        # CompletionTokensDetails and PromptTokensDetails are not serializable and not needed for caching
        for reply in response['replies']:
            if 'meta' in reply and 'usage' in reply['meta']:
                if 'completion_tokens_details' in reply['meta']['usage']:
                    del reply['meta']['usage']['completion_tokens_details']
                if 'prompt_tokens_details' in reply['meta']['usage']:
                    del reply['meta']['usage']['prompt_tokens_details']

        model_cache_dir = self._get_model_cache_dir()
        cache_file = model_cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(response, f)
    
    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        *,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        tools_strict: Optional[bool] = None,
    ):
        """
        Run the chat generator with caching.

        :param messages: List of chat messages
        :return: The generated response
        """
        cache_key = self._get_cache_key(messages, generation_kwargs)
        cached_response = self._get_cached_response(cache_key)
        
        if cached_response is not None:
            return cached_response
        
        # If not in cache, generate response using parent class
        response = super(CachedOpenAIChatGenerator, self).run(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict
        )
        
        # Cache the response
        self._cache_response(cache_key, response.copy())
        
        return response
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the component's configuration to a dictionary.

        :return: A dictionary containing the component's configuration parameters.
        """
        return default_to_dict(
            self,
            cache_dir=str(self.cache_dir),
            **super(CachedOpenAIChatGenerator, self).to_dict()
        )
