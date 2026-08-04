import requests


class CleanupError(Exception):
    pass


def build_cleanup_prompt(raw_text: str) -> str:
    return (
        "Clean up this raw speech-to-text transcript: remove filler words "
        "(um, uh, ıı, şey), add proper punctuation and capitalization, and "
        "fix words that are clearly misheard given the context. Keep the "
        "response in the same language as the transcript -- never translate "
        "it. Return only the cleaned transcript, nothing else.\n\n"
        f"Raw transcript: {raw_text}\n\nCleaned transcript:"
    )


def clean_transcript(raw_text: str, config: dict) -> str:
    url = f"{config['ollama_url']}/api/generate"
    payload = {
        "model": config["ollama_model"],
        "prompt": build_cleanup_prompt(raw_text),
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise CleanupError(f"Ollama request failed: {exc}") from exc
    except ValueError as exc:
        raise CleanupError(f"Ollama returned invalid JSON: {exc}") from exc

    text = data.get("response")
    if not text:
        raise CleanupError("Ollama response missing 'response' field")
    return text.strip()
