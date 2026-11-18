import pytest

from chat_moderation.preprocessing import (
    canonicalize_text,
    normalize_turkish_text,
    reduce_repeated_characters,
    tokenize_turkish_text,
    turkish_lower,
)


def test_turkish_lower_uses_turkish_casing() -> None:
    assert turkish_lower("IĞDIR İZMİR") == "ığdır izmir"


def test_canonicalize_text_normalizes_case_and_whitespace() -> None:
    result = canonicalize_text("  MERHABA&nbsp;\nDÜNYA  ")

    assert result == "merhaba dünya"


def test_normalize_text_replaces_urls_and_mentions() -> None:
    result = normalize_turkish_text(
        "@Berke BAK https://example.com/test"
    )

    assert result == "user bak url"


def test_reduce_repeated_characters_limits_repetitions() -> None:
    result = reduce_repeated_characters(
        "çoookkk güzeeel!!!",
        max_repeats=2,
    )

    assert result == "çookk güzeel!!"


def test_tokenize_turkish_text_returns_normalized_tokens() -> None:
    result = tokenize_turkish_text("@USER ÇOOOK iyi! 123")

    assert result == ["user", "çook", "iyi", "123"]


def test_reduce_repeated_characters_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        reduce_repeated_characters("test", max_repeats=0)