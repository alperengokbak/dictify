import requests


class CleanupError(Exception):
    pass


STYLE_INSTRUCTIONS = {
    "professional": (
        "8. Additionally, rewrite this into a polished, professional tone "
        "suitable for a work email or report: remove casual hedging like "
        "'yeah'/'probably'/'just', expand contractions, and use precise, "
        "businesslike language. Do not add new information or change the "
        "meaning.\n"
        "   Example: \"so yeah I think we should probably just ship it "
        "today, it's good enough\" becomes \"I recommend we proceed with "
        "shipping it today, as it meets the required standard.\"\n"
    ),
    "casual": (
        "8. Additionally, keep the tone relaxed and conversational, very "
        "close to how it was actually spoken - do not formalize the "
        "wording or strip out personality, just apply the cleanup above.\n"
    ),
}


def build_cleanup_prompt(
    raw_text: str, glossary: list[str] | None = None, style: str = "default"
) -> str:
    glossary_section = ""
    if glossary:
        glossary_section = (
            "Known terms/names - if a word in the transcript is a misheard "
            "version of one of these, fix it to match exactly. Do not use "
            "these to change words that are already correct or unrelated:\n"
            f"{', '.join(glossary)}\n\n"
        )
    style_instruction = STYLE_INSTRUCTIONS.get(style, "")
    return (
        "**CRITICAL: DO NOT TRANSLATE. RESPOND ONLY IN THE ORIGINAL LANGUAGE OF THE INPUT.**\n"
        "**NEVER OUTPUT ENGLISH FOR TURKISH INPUT. NEVER OUTPUT ENGLISH FOR ANY NON-ENGLISH INPUT.**\n\n"
        "Your task: Clean up a raw speech-to-text transcript.\n"
        "Instructions:\n"
        "1. Remove verbal fillers and hesitation sounds people use while "
        "thinking - words like um, uh, er, like, you know, I mean (English) "
        "and ıı, ee, şey, yani, hani, işte (Turkish). Only strip these when "
        "they are being used as filler. If the same word is doing real work "
        "in the sentence (e.g. 'şey' meaning an actual 'thing', not a "
        "hesitation sound), keep it - do not blanket-remove it.\n"
        "2. Add punctuation and capitalization where appropriate\n"
        "3. Fix clearly misheard words based on context\n"
        "4. Fix grammar mistakes (verb tense, subject-verb agreement, word "
        "order, articles, etc.) without changing the meaning or the "
        "language of the input\n"
        "5. DO NOT add or remove content beyond cleaning\n"
        "6. DO NOT TRANSLATE - output MUST be in the same language as input\n"
        "7. Output ONLY the cleaned transcript text\n"
        f"{style_instruction}\n"
        "EXAMPLES:\n"
        "Input: 'ıı bugün şey çalıştım yani'\n"
        "Output: 'Bugün çalıştım.'\n"
        "(Turkish → Turkish, fillers removed, not translated)\n\n"
        "Input: 'so um I think, uh, we should ship it'\n"
        "Output: 'I think we should ship it.'\n"
        "(English → English, fillers removed)\n\n"
        "Input: 'I have went to the store yesterday and I buyed some milk'\n"
        "Output: 'I went to the store yesterday and I bought some milk.'\n"
        "(grammar fixed, meaning and language unchanged)\n\n"
        f"{glossary_section}"
        f"TRANSCRIPT TO CLEAN:\n"
        f"---\n{raw_text}\n---\n\n"
        "CLEANED TRANSCRIPT (same language as input, no translation):"
    )


def clean_transcript(raw_text: str, config: dict) -> str:
    url = f"{config['ollama_url']}/api/generate"
    payload = {
        "model": config["ollama_model"],
        "prompt": build_cleanup_prompt(
            raw_text,
            glossary=config.get("glossary"),
            style=config.get("style", "default"),
        ),
        "stream": False,
        "keep_alive": "30m",
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

    try:
        text = data.get("response")
        if not text:
            raise CleanupError("Ollama response missing 'response' field")
        text = text.strip()
    except (AttributeError, TypeError) as exc:
        raise CleanupError(f"Ollama returned unexpected response shape: {exc}") from exc
    return text
