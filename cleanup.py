import requests


class CleanupError(Exception):
    pass


# Maps Whisper's detected language codes to names for the prompt anchor
# below. "auto"/"unknown"/unrecognized codes fall back to no anchor at all
# rather than a wrong or confusing one.
LANGUAGE_NAMES = {"en": "English", "tr": "Turkish"}

# Matching wrapping-quote pairs the cleanup model has been observed to add
# around its whole answer (straight and curly, single and double).
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’"}


def _strip_added_wrapping_quotes(cleaned: str, raw_text: str) -> str:
    """The cleanup model sometimes wraps its entire answer in quotation
    marks that were never in the raw Whisper transcript - observed live
    (2026-08-09) via history.jsonl, e.g. "Hello there" comes back as
    '"Hello there."'. The prompt now asks it not to, but small local models
    don't reliably follow every instruction, so also strip a wrapping pair
    defensively. Only strips when the raw transcript did NOT already start
    and end with the same quote character, so genuinely quoted speech in
    the original dictation is left untouched."""
    if len(cleaned) < 2:
        return cleaned
    open_q, close_q = cleaned[0], cleaned[-1]
    if _QUOTE_PAIRS.get(open_q) != close_q:
        return cleaned
    raw_stripped = raw_text.strip()
    if raw_stripped[:1] == open_q and raw_stripped[-1:] == close_q:
        return cleaned
    return cleaned[1:-1].strip()

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
    raw_text: str,
    glossary: list[str] | None = None,
    style: str = "default",
    language: str | None = None,
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
    # Whisper already knows the spoken language for certain - anchor that
    # fact explicitly instead of leaving the model to infer it from the
    # text alone, which has been observed to drift wrong on long/rambling
    # input even with the "DO NOT TRANSLATE" instruction already present.
    language_name = LANGUAGE_NAMES.get(language)
    language_anchor = (
        f"**The speech recognizer detected this transcript's spoken "
        f"language as: {language_name}. Your output MUST be in "
        f"{language_name} - do not switch languages under any "
        f"circumstances.**\n"
        if language_name
        else ""
    )
    return (
        "**CRITICAL: DO NOT TRANSLATE. RESPOND ONLY IN THE ORIGINAL LANGUAGE OF THE INPUT.**\n"
        "**NEVER OUTPUT ENGLISH FOR TURKISH INPUT. NEVER OUTPUT ENGLISH FOR ANY NON-ENGLISH INPUT.**\n"
        f"{language_anchor}\n"
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
        "5. DO NOT add or remove content beyond cleaning. Every statement in "
        "the transcript must survive into your output. Never drop, summarize "
        "or condense a sentence or clause - not even one that sounds like "
        "background context, an aside, or a lead-in to the real point. Only "
        "the filler words in rule 1 get removed; complete thoughts never do.\n"
        "6. DO NOT TRANSLATE - output MUST be in the same language as input\n"
        "7. Output ONLY the cleaned transcript text - do not wrap it in "
        "quotation marks and do not add labels or commentary\n"
        f"{style_instruction}\n"
        "EXAMPLES:\n"
        # Every other example here shows the output coming back shorter than
        # the input, which a 3B model generalises into "shorter is better" and
        # then applies to whole sentences. This one demonstrates the opposite
        # case explicitly - see rule 5.
        "Input: 'and if that doesn't work then we can just revert it back to "
        "the old one which was fine am I right'\n"
        "Output: 'And if that doesn't work, then we can revert it back to the "
        "old one, which was fine. Am I right?'\n"
        "(every clause kept - the context sentence is NOT dropped, only "
        "punctuation and capitalization added)\n\n"
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


def clean_transcript(raw_text: str, config: dict, language: str | None = None) -> str:
    url = f"{config['ollama_url']}/api/generate"
    payload = {
        "model": config["ollama_model"],
        "prompt": build_cleanup_prompt(
            raw_text,
            glossary=config.get("glossary"),
            style=config.get("style", "default"),
            language=language,
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
    return _strip_added_wrapping_quotes(text, raw_text)
