"""
Utility functions for output parsing and format normalization across Shankh agents and MCP servers.
"""

import json
from pathlib import Path
from typing import Any, List


def load_prompt(prompt_name: str) -> str:
    """Load a prompt from config/prompts by name."""
    project_root = Path(__file__).resolve().parents[3]
    possible_paths = [
        project_root / "config" / "prompts" / f"{prompt_name}.md",
        project_root / "config" / "prompts" / f"{prompt_name}.txt",
        Path("config/prompts") / f"{prompt_name}.md",
    ]

    for path in possible_paths:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()

    raise FileNotFoundError(
        f"Prompt file '{prompt_name}' not found. Searched in: {[str(p) for p in possible_paths]}"
    )


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
    """Extract text from the last message. The last message is always the agent's final response."""
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", last)
    return extract_text_content(content)
