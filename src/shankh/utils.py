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
