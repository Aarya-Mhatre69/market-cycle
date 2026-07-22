"""
Utility functions for output parsing and format normalization across Shankh agents and MCP servers.
"""

import json
from typing import Any, List


def extract_text_content(content: Any) -> str:
    """
    Extract plain string text from LLM response content.
    Handles raw strings, lists of content blocks (e.g. [{'type': 'text', 'text': '...'}]),
    dictionaries, None, and arbitrary objects.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
                elif "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        if content.get("type") == "text" and "text" in content:
            return str(content["text"])
        elif "text" in content:
            return str(content["text"])
        return json.dumps(content)
    if content is None:
        return ""
    return str(content)


def extract_response_text(messages: List[Any]) -> str:
    """
    Extract non-empty text content from the latest message in a message list.
    Searches backwards from the last message to find the first message with non-empty text.
    """
    if not messages:
        return ""
    for msg in reversed(messages):
        content = getattr(msg, "content", msg)
        text = extract_text_content(content)
        if text.strip():
            return text
    return ""


def resolve_model(provider_hint: str = None) -> Any:
    """
    Centralized LLM model resolver for Shankh agents.
    Prioritizes available API keys in environment:
    1. Provider hint / specific keys (Cerebras, MistralAI, Google)
    2. Fallback to Gemini 2.5 Flash if Google key is configured
    """
    import os

    if provider_hint == "cerebras" and os.getenv("CEREBRAS_API_KEY"):
        from langchain_cerebras import ChatCerebras
        return ChatCerebras(model="llama-3.3-70b", api_key=os.getenv("CEREBRAS_API_KEY"))

    if provider_hint == "mistral" and os.getenv("MISTRALAI_API_KEY"):
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model="mistral-large-latest", api_key=os.getenv("MISTRALAI_API_KEY"))

    if os.getenv("GOOGLE_API_KEY"):
        return "google_genai:gemini-2.5-flash"

    if os.getenv("CEREBRAS_API_KEY"):
        from langchain_cerebras import ChatCerebras
        return ChatCerebras(model="llama-3.3-70b", api_key=os.getenv("CEREBRAS_API_KEY"))

    if os.getenv("MISTRALAI_API_KEY"):
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model="mistral-large-latest", api_key=os.getenv("MISTRALAI_API_KEY"))

    # Return default string format for LangChain standard provider binding
    return "google_genai:gemini-2.5-flash"

