from unittest.mock import Mock, patch

import pytest

import cleanup

CONFIG = {"ollama_url": "http://localhost:11434", "ollama_model": "qwen2.5:3b"}


def test_build_cleanup_prompt_includes_raw_text_and_language_instruction():
    prompt = cleanup.build_cleanup_prompt("ıı bugün şey kubernetis üzerinde çalıştım")
    assert "ıı bugün şey kubernetis üzerinde çalıştım" in prompt
    assert "never translate" in prompt


@patch("cleanup.requests.post")
def test_clean_transcript_sends_correct_request_and_parses_response(mock_post):
    mock_post.return_value = Mock(
        json=lambda: {"response": " Bugün Kubernetes üzerinde çalıştım. "}
    )
    mock_post.return_value.raise_for_status = lambda: None

    result = cleanup.clean_transcript(
        "ıı bugün şey kubernetis üzerinde çalıştım", CONFIG
    )

    assert result == "Bugün Kubernetes üzerinde çalıştım."
    called_args, called_kwargs = mock_post.call_args
    assert called_args[0] == "http://localhost:11434/api/generate"
    assert called_kwargs["json"]["model"] == "qwen2.5:3b"
    assert called_kwargs["json"]["stream"] is False


@patch("cleanup.requests.post", side_effect=cleanup.requests.exceptions.ConnectionError("refused"))
def test_clean_transcript_raises_cleanup_error_on_connection_failure(mock_post):
    with pytest.raises(cleanup.CleanupError):
        cleanup.clean_transcript("some text", CONFIG)


@patch("cleanup.requests.post")
def test_clean_transcript_raises_cleanup_error_on_missing_response_field(mock_post):
    mock_post.return_value = Mock(json=lambda: {"done": True})
    mock_post.return_value.raise_for_status = lambda: None

    with pytest.raises(cleanup.CleanupError):
        cleanup.clean_transcript("some text", CONFIG)
