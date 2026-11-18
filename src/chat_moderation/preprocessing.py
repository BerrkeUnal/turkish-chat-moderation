from __future__ import annotations

import html
import re
import unicodedata

TURKISH_CASE_TRANSLATION = str.maketrans(
    {
        "I": "ı",
        "İ": "i",
    }
)

LEETSPEAK_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)

OBFUSCATION_SEPARATOR_PATTERN = re.compile(
    r"(?<=[a-zçğıöşü])[._*+\-]+(?=[a-zçğıöşü])",
    re.IGNORECASE,
)

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
USER_MENTION_PATTERN = re.compile(r"(?<![\w@])@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[0-9a-zçğıöşü]+", re.IGNORECASE)


def turkish_lower(text: str) -> str:
    """Convert text to lowercase using Turkish casing rules."""

    return text.translate(TURKISH_CASE_TRANSLATION).lower()


def canonicalize_text(text: str) -> str:
    """Create a stable Turkish-aware text key."""

    normalized = unicodedata.normalize("NFKC", html.unescape(str(text)))
    normalized = turkish_lower(normalized)

    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def reduce_repeated_characters(
    text: str,
    max_repeats: int = 2,
) -> str:
    """Limit consecutive identical characters."""

    if max_repeats < 1:
        raise ValueError("max_repeats must be at least 1.")

    pattern = re.compile(rf"(.)\1{{{max_repeats},}}", re.DOTALL)

    return pattern.sub(
        lambda match: match.group(1) * max_repeats,
        text,
    )


def normalize_turkish_text(
    text: str | None,
    *,
    lowercase: bool = True,
    replace_urls: bool = True,
    replace_user_mentions: bool = True,
    max_repeated_characters: int = 2,
) -> str:
    """Normalize Turkish social-media text."""

    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", html.unescape(str(text)))

    if replace_urls:
        normalized = URL_PATTERN.sub(" URL ", normalized)

    if replace_user_mentions:
        normalized = USER_MENTION_PATTERN.sub(" USER ", normalized)

    if lowercase:
        normalized = turkish_lower(normalized)

    normalized = reduce_repeated_characters(
        normalized,
        max_repeats=max_repeated_characters,
    )

    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalize_for_lexicon(text: str | None) -> str:
    """Normalize common offensive-word obfuscations."""

    normalized = normalize_turkish_text(text, lowercase=True)
    normalized = normalized.translate(LEETSPEAK_TRANSLATION)
    normalized = OBFUSCATION_SEPARATOR_PATTERN.sub("", normalized)

    return normalized


def tokenize_for_lexicon(text: str | None) -> list[str]:
    """Tokenize text after leetspeak and obfuscation normalization."""

    return TOKEN_PATTERN.findall(normalize_for_lexicon(text))


def tokenize_turkish_text(
    text: str | None,
    *,
    max_repeated_characters: int = 2,
) -> list[str]:
    """Normalize and tokenize text for classical and neural models."""

    normalized = normalize_turkish_text(
        text,
        lowercase=True,
        max_repeated_characters=max_repeated_characters,
    )

    return TOKEN_PATTERN.findall(normalized)