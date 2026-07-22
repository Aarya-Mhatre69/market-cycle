import pytest
from langchain_core.messages import AIMessage, HumanMessage
from shankh.utils import extract_response_text, extract_text_content


def test_extract_text_content_string():
    assert extract_text_content("Hello World") == "Hello World"


def test_extract_text_content_list_of_dicts():
    content = [{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}]
    assert extract_text_content(content) == "Part 1\nPart 2"


def test_extract_text_content_list_of_strings():
    content = ["Hello", "World"]
    assert extract_text_content(content) == "Hello\nWorld"


def test_extract_text_content_dict():
    content = {"type": "text", "text": "Structured text"}
    assert extract_text_content(content) == "Structured text"


def test_extract_text_content_none():
    assert extract_text_content(None) == ""


def test_extract_response_text_from_messages():
    messages = [
        HumanMessage(content="What is the market snapshot?"),
        AIMessage(content=[{"type": "text", "text": "Market is strong today."}]),
    ]
    extracted = extract_response_text(messages)
    assert isinstance(extracted, str)
    assert extracted == "Market is strong today."


def test_extract_response_text_fallback_to_previous():
    messages = [
        HumanMessage(content="Hi"),
        AIMessage(content="Valid response"),
        AIMessage(content=""),
    ]
    extracted = extract_response_text(messages)
    assert extracted == "Valid response"
