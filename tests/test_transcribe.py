from pathlib import Path

import transcribe

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_whisper_json_single_segment():
    text, lang = transcribe._parse_whisper_json(str(FIXTURES / "whisper_output_en.json"))
    assert text == "Hello, this is a test of the transcription pipeline."
    assert lang == "en"


def test_parse_whisper_json_joins_multiple_segments():
    text, lang = transcribe._parse_whisper_json(str(FIXTURES / "whisper_output_multi.json"))
    assert text == "Bugün Kubernetes üzerinde çalıştım."
    assert lang == "tr"
