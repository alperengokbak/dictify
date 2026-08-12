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


def test_build_cleanup_prompt_includes_grammar_correction_instruction():
    prompt = cleanup.build_cleanup_prompt("some text")
    assert "grammar" in prompt.lower()
    # the worked example anchors what "grammar" means here, not just the
    # instruction's own wording
    assert "I have went to the store" in prompt
    assert "I went to the store" in prompt


def test_build_cleanup_prompt_anchors_detected_language_when_known():
    # Regression: observed live (2026-08-06) - the model drifted into
    # Turkish, or garbled both languages together, on long/rambling
    # English input, despite the "DO NOT TRANSLATE" instruction already
    # present. history.jsonl showed Whisper had detected "en" correctly
    # both times - the language just never reached the cleanup prompt,
    # leaving a small local model to infer it from content alone. Anchor
    # the already-known language explicitly instead of relying on
    # inference.
    prompt = cleanup.build_cleanup_prompt("some text", language="en")
    assert "English" in prompt
    assert "MUST be in English" in prompt


def test_build_cleanup_prompt_maps_tr_code_to_turkish_name():
    prompt = cleanup.build_cleanup_prompt("some text", language="tr")
    assert "Turkish" in prompt
    assert "MUST be in Turkish" in prompt


def test_build_cleanup_prompt_omits_language_anchor_when_language_unknown():
    for unknown_language in (None, "auto", "unknown"):
        prompt = cleanup.build_cleanup_prompt("some text", language=unknown_language)
        assert "speech recognizer detected" not in prompt


@patch("cleanup.requests.post")
def test_clean_transcript_passes_detected_language_into_prompt(mock_post):
    mock_post.return_value = Mock(json=lambda: {"response": "cleaned"})
    mock_post.return_value.raise_for_status = lambda: None

    cleanup.clean_transcript("some text", CONFIG, language="en")

    _called_args, called_kwargs = mock_post.call_args
    sent_prompt = called_kwargs["json"]["prompt"]
    assert "MUST be in English" in sent_prompt


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


def test_build_cleanup_prompt_instructs_no_wrapping_quotation_marks():
    prompt = cleanup.build_cleanup_prompt("some text")
    assert "quotation mark" in prompt.lower()


def test_build_cleanup_prompt_forbids_dropping_whole_sentences():
    # Regression: observed live (2026-08-12) via history.jsonl - 8 of 176
    # real dictations came back with entire clauses or sentences missing,
    # e.g. "And if the latest version does not work properly, then we can
    # execute git revert ... Am I right?" was pasted as just "Am I right?".
    # Replaying those transcripts through the real model reproduced it
    # 3/3 times each, so it is a deterministic prompt-following failure,
    # not sampling noise. The bare "DO NOT add or remove content beyond
    # cleaning" was too weak to counteract the shortening pattern the
    # worked examples demonstrate - qwen2.5:3b generalised it into
    # "drop the context and keep the point".
    prompt = cleanup.build_cleanup_prompt("some text").lower()
    assert "summarize" in prompt or "condense" in prompt
    # the failure mode is specifically dropping background/lead-in clauses,
    # so the instruction has to name that case rather than stay generic
    assert "context" in prompt
    assert "clause" in prompt


def test_build_cleanup_prompt_includes_content_preservation_example():
    # Strengthening the instruction alone did not hold (a rephrased rule 5
    # without this example still dropped a sentence in 4 of the 12 replayed
    # cases, including a Turkish one). Every other worked example in the
    # prompt shows the output getting *shorter* than the input; this one
    # exists to demonstrate the opposite - a multi-clause transcript coming
    # back whole - so the examples stop teaching "shorter is better".
    prompt = cleanup.build_cleanup_prompt("some text")
    assert "Am I right?" in prompt
    assert "every clause kept" in prompt.lower()


@patch("cleanup.requests.post")
def test_clean_transcript_strips_quotes_the_model_added(mock_post):
    # Regression: observed live (2026-08-09) via history.jsonl - the
    # cleanup model sometimes wraps its whole answer in quotation marks
    # that were never in the raw Whisper transcript, e.g. "Hello there"
    # comes back as '"Hello there."'. 4 of 71 real dictations hit this.
    mock_post.return_value = Mock(json=lambda: {"response": '"Hello there."'})
    mock_post.return_value.raise_for_status = lambda: None

    result = cleanup.clean_transcript("Hello there", CONFIG)

    assert result == "Hello there."


@patch("cleanup.requests.post")
def test_clean_transcript_keeps_quotes_already_present_in_raw_text(mock_post):
    # If the user's own dictation was genuinely quoted speech, the quotes
    # are real content - only strip a pair the model itself introduced.
    mock_post.return_value = Mock(json=lambda: {"response": '"Hello there."'})
    mock_post.return_value.raise_for_status = lambda: None

    result = cleanup.clean_transcript('"Hello there"', CONFIG)

    assert result == '"Hello there."'


@patch("cleanup.requests.post")
def test_clean_transcript_leaves_unquoted_response_unchanged(mock_post):
    mock_post.return_value = Mock(json=lambda: {"response": "Hello there."})
    mock_post.return_value.raise_for_status = lambda: None

    result = cleanup.clean_transcript("Hello there", CONFIG)

    assert result == "Hello there."


@patch("cleanup.requests.post")
def test_clean_transcript_does_not_strip_a_single_leading_quote(mock_post):
    # Not a wrapping pair (no matching closing quote) - must be left alone
    # rather than mangled by a naive first/last-char strip.
    mock_post.return_value = Mock(json=lambda: {"response": '"He said hi'})
    mock_post.return_value.raise_for_status = lambda: None

    result = cleanup.clean_transcript("He said hi", CONFIG)

    assert result == '"He said hi'
