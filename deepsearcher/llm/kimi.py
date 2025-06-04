import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class Kimi(BaseLLM):
    """
    Kimi language model implementation.

    This class provides an interface to interact with Kimi's language models
    through their API, which is compatible with OpenAI's API.

    API Documentation: https://platform.moonshot.cn/docs
    """

    def __init__(self, model: str = "moonshot-v1-8k", **kwargs):
        """
        Initialize a Kimi language model client.

        Args:
            model (str, optional): The Kimi model identifier to use. Defaults to "moonshot-v1-8k".
                                   Other options include "moonshot-v1-32k", "moonshot-v1-128k",
                                   "moonshot-v1-8k-vision-preview", "moonshot-v1-32k-vision-preview", "moonshot-v1-128k-vision-preview".
            **kwargs: Additional keyword arguments to pass to the OpenAI client.
                - api_key: Kimi API key. If not provided, uses MOONSHOT_API_KEY environment variable.
                - base_url: Kimi API base URL. If not provided, uses MOONSHOT_BASE_URL environment
                  variable or defaults to "https://api.moonshot.cn/v1".
        """
        super().__init__()
        from openai import OpenAI as OpenAI_

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("MOONSHOT_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = kwargs.pop("base_url", os.getenv("MOONSHOT_BASE_URL", default="https://api.moonshot.cn/v1"))
        self.client = OpenAI_(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict], **kwargs) -> ChatResponse:
        """
        Send a chat message to the Kimi model and get a response.

        Args:
            messages (List[Dict]): A list of message dictionaries, typically in the format
                                  [{"role": "system", "content": "..."},
                                   {"role": "user", "content": "..."}]
            **kwargs: Additional keyword arguments to pass to the underlying OpenAI API call,
                      e.g., 'response_format', 'tools', 'tool_choice'.

        Returns:
            ChatResponse: An object containing the model's response and token usage information.
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        if hasattr(completion.choices[0].message, 'tool_calls') and completion.choices[0].message.tool_calls:
            tool_calls_content = str(completion.choices[0].message.tool_calls)
            return ChatResponse(
                content=tool_calls_content,
                total_tokens=completion.usage.total_tokens,
            )
        else:
            return ChatResponse(
                content=completion.choices[0].message.content,
                total_tokens=completion.usage.total_tokens,
            )