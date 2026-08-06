from unittest.mock import Mock, patch

import pytest

import cleanup

CONFIG = {"ollama_url": "http://localhost:11434", "ollama_model": "qwen2.5:3b"}


def test_build_cleanup_prompt_includes_raw_text_and_language_instruction():
    prompt = cleanup.build_cleanup_prompt("ıı bugün şey kubernetis üzerinde çalıştım")
    assert "ıı bugün şey kubernetis üzerinde çalıştım" in prompt
    assert "DO NOT TRANSLATE" in prompt


def test_build_cleanup_prompt_omits_glossary_section_when_empty():
    prompt = cleanup.build_cleanup_prompt("some text")
    assert "Known terms" not in prompt


def test_build_cleanup_prompt_includes_glossary_terms_when_provided():
    prompt = cleanup.build_cleanup_prompt(
        "some text", glossary=["Kubernetes", "PyQt", "Grafana"]
    )
    assert "Kubernetes" in prompt
    assert "PyQt" in prompt
    assert "Grafana" in prompt


@patch("cleanup.requests.post")
def test_clean_transcript_passes_configured_glossary_into_prompt(mock_post):
    mock_post.return_value = Mock(json=lambda: {"response": "cleaned"})
    mock_post.return_value.raise_for_status = lambda: None

    config_with_glossary = dict(CONFIG, glossary=["Kubernetes", "PyQt"])
    cleanup.clean_transcript("some text", config_with_glossary)

    _called_args, called_kwargs = mock_post.call_args
    sent_prompt = called_kwargs["json"]["prompt"]
    assert "Kubernetes" in sent_prompt
    assert "PyQt" in sent_prompt


def test_build_cleanup_prompt_default_style_has_no_tone_instruction():
    prompt = cleanup.build_cleanup_prompt("some text")
    assert "professional tone" not in prompt
    assert "relaxed and conversational" not in prompt


def test_build_cleanup_prompt_professional_style_adds_tone_instruction():
    prompt = cleanup.build_cleanup_prompt("some text", style="professional")
    assert "professional tone" in prompt


def test_build_cleanup_prompt_casual_style_adds_tone_instruction():
    prompt = cleanup.build_cleanup_prompt("some text", style="casual")
    assert "relaxed and conversational" in prompt


@patch("cleanup.requests.post")
def test_clean_transcript_passes_configured_style_into_prompt(mock_post):
    mock_post.return_value = Mock(json=lambda: {"response": "cleaned"})
    mock_post.return_value.raise_for_status = lambda: None

    config_with_style = dict(CONFIG, style="professional")
    cleanup.clean_transcript("some text", config_with_style)

    _called_args, called_kwargs = mock_post.call_args
    sent_prompt = called_kwargs["json"]["prompt"]
    assert "professional tone" in sent_prompt


def test_build_cleanup_prompt_lists_filler_words_in_both_languages():
    prompt = cleanup.build_cleanup_prompt("some text")
    # English hesitation fillers
    for filler in ("um", "uh", "you know"):
        assert filler in prompt
    # Turkish hesitation fillers
    for filler in ("ıı", "şey", "yani", "hani"):
        assert filler in prompt
    # must not blanket-strip these words when they carry real meaning
    assert "only" in prompt.lower() or "real meaning" in prompt.lower()


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


@patch("cleanup.requests.post")
def test_clean_transcript_raises_cleanup_error_on_non_dict_response(mock_post):
    # Ollama is expected to return a JSON object, but if it ever returns
    # something else (e.g. a bare list), .get() must not escape as a
    # raw AttributeError - it should surface as CleanupError like any
    # other malformed-response case, so the caller's fallback path runs.
    mock_post.return_value = Mock(json=lambda: ["unexpected", "list", "response"])
    mock_post.return_value.raise_for_status = lambda: None

    with pytest.raises(cleanup.CleanupError):
        cleanup.clean_transcript("some text", CONFIG)


@patch("cleanup.requests.post")
def test_clean_transcript_sets_keep_alive_to_avoid_cold_reload(mock_post):
    mock_post.return_value = Mock(json=lambda: {"response": "cleaned"})
    mock_post.return_value.raise_for_status = lambda: None

    cleanup.clean_transcript("some text", CONFIG)

    _called_args, called_kwargs = mock_post.call_args
    assert called_kwargs["json"]["keep_alive"] == "30m"
