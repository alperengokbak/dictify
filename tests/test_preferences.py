import preferences


def test_parse_glossary_text_splits_lines_and_strips_empties():
    result = preferences._parse_glossary_text("Kubernetes\n  PyQt  \n\nGrafana\n")
    assert result == ["Kubernetes", "PyQt", "Grafana"]


def test_parse_glossary_text_empty_input_returns_empty_list():
    assert preferences._parse_glossary_text("") == []
    assert preferences._parse_glossary_text("   \n  \n") == []


def test_format_glossary_text_joins_with_newlines():
    assert preferences._format_glossary_text(["Kubernetes", "PyQt"]) == "Kubernetes\nPyQt"


def test_format_glossary_text_empty_list_returns_empty_string():
    assert preferences._format_glossary_text([]) == ""


def test_parse_float_or_default_valid_input():
    assert preferences._parse_float_or_default("-55.0", 0.0) == -55.0
    assert preferences._parse_float_or_default("  10.5  ", 0.0) == 10.5


def test_parse_float_or_default_invalid_input_returns_default():
    assert preferences._parse_float_or_default("not a number", -55.0) == -55.0
    assert preferences._parse_float_or_default("", -55.0) == -55.0


def test_parse_int_or_default_valid_input():
    assert preferences._parse_int_or_default("200", 0) == 200
    assert preferences._parse_int_or_default("  42  ", 0) == 42


def test_parse_int_or_default_invalid_input_returns_default():
    assert preferences._parse_int_or_default("abc", 200) == 200
    assert preferences._parse_int_or_default("12.5", 200) == 200
    assert preferences._parse_int_or_default("", 200) == 200
