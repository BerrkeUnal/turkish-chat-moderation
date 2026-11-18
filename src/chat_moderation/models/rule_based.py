from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from rapidfuzz.distance import Levenshtein

from chat_moderation.models.base import BaseTextClassifier, Prediction
from chat_moderation.preprocessing import tokenize_for_lexicon


@dataclass(frozen=True)
class LexiconMatch:
    """One lexicon match found in a message."""

    token: str
    lexicon_term: str
    edit_distance: int
    similarity: float
    exact: bool


class RuleBasedClassifier(BaseTextClassifier):
    """Offensive-language baseline using a fuzzy lexicon."""

    def __init__(self, lexicon_path: str | Path) -> None:
        self.lexicon_path = Path(lexicon_path)
        self.lexicon_terms = self._load_lexicon(self.lexicon_path)
        self.lexicon_set = set(self.lexicon_terms)

    @property
    def model_name(self) -> str:
        return "Rule-Based Edit Distance"

    @staticmethod
    def _load_lexicon(lexicon_path: Path) -> tuple[str, ...]:
        if not lexicon_path.exists():
            raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")

        terms: set[str] = set()

        with lexicon_path.open("r", encoding="utf-8") as lexicon_file:
            for raw_line in lexicon_file:
                line = raw_line.strip()

                if not line or line.startswith("#"):
                    continue

                tokens = tokenize_for_lexicon(line)

                if len(tokens) != 1:
                    raise ValueError(
                        "Every lexicon entry must contain exactly one token: "
                        f"{line!r}"
                    )

                terms.add(tokens[0])

        if not terms:
            raise ValueError("The offensive lexicon is empty.")

        return tuple(sorted(terms))

    @staticmethod
    def _allowed_edit_distance(token: str, term: str) -> int:
        minimum_length = min(len(token), len(term))
        maximum_length = max(len(token), len(term))

        if minimum_length < 4:
            return 0

        if maximum_length <= 6:
            return 1

        return 2

    def _find_fuzzy_match(self, token: str) -> LexiconMatch | None:
        best_term: str | None = None
        best_distance: int | None = None

        for term in self.lexicon_terms:
            allowed_distance = self._allowed_edit_distance(token, term)

            if allowed_distance == 0:
                continue

            if abs(len(token) - len(term)) > allowed_distance:
                continue

            distance = Levenshtein.distance(token, term)

            if distance > allowed_distance:
                continue

            if best_distance is None or distance < best_distance:
                best_term = term
                best_distance = distance

        if best_term is None or best_distance is None:
            return None

        maximum_length = max(len(token), len(best_term))
        similarity = 1.0 - (best_distance / maximum_length)

        return LexiconMatch(
            token=token,
            lexicon_term=best_term,
            edit_distance=best_distance,
            similarity=round(similarity, 6),
            exact=False,
        )

    def find_matches(self, text: str) -> list[LexiconMatch]:
        """Find exact and edit-distance lexicon matches."""

        matches: list[LexiconMatch] = []

        for token in tokenize_for_lexicon(text):
            if token in self.lexicon_set:
                matches.append(
                    LexiconMatch(
                        token=token,
                        lexicon_term=token,
                        edit_distance=0,
                        similarity=1.0,
                        exact=True,
                    )
                )
                continue

            fuzzy_match = self._find_fuzzy_match(token)

            if fuzzy_match is not None:
                matches.append(fuzzy_match)

        return matches

    def predict(self, text: str) -> Prediction:
        """Classify one message using lexicon matches."""

        start_time = perf_counter()
        matches = self.find_matches(text)

        label = 1 if matches else 0
        offensive_score = max(
            (match.similarity for match in matches),
            default=0.0,
        )

        confidence = offensive_score if label == 1 else 1.0
        latency_ms = (perf_counter() - start_time) * 1000

        return Prediction(
            label=label,
            label_name="offensive" if label == 1 else "clean",
            confidence=round(confidence, 6),
            latency_ms=round(latency_ms, 6),
            metadata={
                "matched_terms": [asdict(match) for match in matches],
                "confidence_type": "heuristic",
            },
        )