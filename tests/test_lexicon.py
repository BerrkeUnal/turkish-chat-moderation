from pathlib import Path

from chat_moderation.models.rule_based import RuleBasedClassifier


def create_classifier(tmp_path: Path) -> RuleBasedClassifier:
    lexicon_path = tmp_path / "offensive_terms.txt"
    lexicon_path.write_text(
        "salak\naptal\ngerizekalı\nit\n",
        encoding="utf-8",
    )

    return RuleBasedClassifier(lexicon_path)


def test_rule_model_detects_exact_match(tmp_path: Path) -> None:
    classifier = create_classifier(tmp_path)

    prediction = classifier.predict("Sen tam bir salak")

    assert prediction.label == 1
    assert prediction.label_name == "offensive"
    assert prediction.confidence == 1.0


def test_rule_model_detects_edit_distance_match(tmp_path: Path) -> None:
    classifier = create_classifier(tmp_path)

    prediction = classifier.predict("salakk")

    assert prediction.label == 1
    assert prediction.metadata["matched_terms"][0]["edit_distance"] == 1


def test_rule_model_detects_leetspeak(tmp_path: Path) -> None:
    classifier = create_classifier(tmp_path)

    prediction = classifier.predict("s4l4k")

    assert prediction.label == 1
    assert prediction.metadata["matched_terms"][0]["exact"] is True


def test_rule_model_detects_obfuscated_word(tmp_path: Path) -> None:
    classifier = create_classifier(tmp_path)

    prediction = classifier.predict("a.p.t.a.l")

    assert prediction.label == 1


def test_rule_model_keeps_clean_message_clean(tmp_path: Path) -> None:
    classifier = create_classifier(tmp_path)

    prediction = classifier.predict("Bugün hava gerçekten çok güzel")

    assert prediction.label == 0
    assert prediction.label_name == "clean"


def test_rule_model_disables_fuzzy_matching_for_short_terms(
    tmp_path: Path,
) -> None:
    classifier = create_classifier(tmp_path)

    prediction = classifier.predict("at")

    assert prediction.label == 0