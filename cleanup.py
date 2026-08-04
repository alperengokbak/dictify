import requests


class CleanupError(Exception):
    pass


def build_cleanup_prompt(raw_text: str) -> str:
    return (
        "**CRITICAL: DO NOT TRANSLATE. RESPOND ONLY IN THE ORIGINAL LANGUAGE OF THE INPUT.**\n"
        "**NEVER OUTPUT ENGLISH FOR TURKISH INPUT. NEVER OUTPUT ENGLISH FOR ANY NON-ENGLISH INPUT.**\n\n"
        "Your task: Clean up raw speech-to-text transcript.\n"
        "Instructions:\n"
        "1. Remove ONLY filler words (um, uh, ıı, şey, yani)\n"
        "2. Add punctuation and capitalization where appropriate\n"
        "3. Fix clearly misheard words based on context\n"
        "4. DO NOT add or remove content beyond cleaning\n"
        "5. DO NOT TRANSLATE - output MUST be in the same language as input\n"
        "6. Output ONLY the cleaned transcript text\n\n"
        "LANGUAGE PRESERVATION EXAMPLE:\n"
        "Input: 'ıı bugün şey çalıştım yani'\n"
        "Output: 'Bugün çalıştım.'\n"
        "(Turkish → Turkish, not Turkish → English)\n\n"
        f"TRANSCRIPT TO CLEAN:\n"
        f"---\n{raw_text}\n---\n\n"
        "CLEANED TRANSCRIPT (same language as input, no translation):"
    )


def clean_transcript(raw_text: str, config: dict) -> str:
    url = f"{config['ollama_url']}/api/generate"
    payload = {
        "model": config["ollama_model"],
        "prompt": build_cleanup_prompt(raw_text),
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
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
