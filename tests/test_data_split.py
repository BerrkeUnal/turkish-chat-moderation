import pandas as pd
import pytest

from chat_moderation.data import clean_training_pool, validate_split_ids


def create_id_frame(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"id": ids})


def create_labeled_frame(
    rows: list[tuple[str, str, int]],
) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["id", "text", "label"])


def test_validate_split_ids_accepts_disjoint_splits() -> None:
    splits = {
        "train": create_id_frame(["1", "2", "3"]),
        "validation": create_id_frame(["4", "5"]),
        "test": create_id_frame(["6", "7"]),
    }

    validate_split_ids(splits)


def test_validate_split_ids_rejects_overlapping_splits() -> None:
    splits = {
        "train": create_id_frame(["1", "2", "3"]),
        "validation": create_id_frame(["3", "4"]),
        "test": create_id_frame(["5", "6"]),
    }

    with pytest.raises(ValueError, match="ID overlap"):
        validate_split_ids(splits)


def test_clean_training_pool_removes_duplicates_and_test_overlap() -> None:
    training_frame = create_labeled_frame(
        [
            ("1", "Merhaba dünya", 0),
            ("2", "  MERHABA DÜNYA  ", 0),
            ("3", "Test mesajı", 0),
        ]
    )
    test_frame = create_labeled_frame(
        [
            ("4", "TEST MESAJI", 0),
        ]
    )

    cleaned_frame, statistics = clean_training_pool(
        training_frame=training_frame,
        test_frame=test_frame,
        text_column="text",
        label_column="label",
    )

    assert cleaned_frame["id"].tolist() == ["1"]
    assert statistics["removed_duplicate_rows"] == 1
    assert statistics["removed_test_overlap_rows"] == 1
    assert statistics["final_training_pool_rows"] == 1


def test_clean_training_pool_removes_ambiguous_texts() -> None:
    training_frame = create_labeled_frame(
        [
            ("1", "same text", 0),
            ("2", "SAME TEXT", 1),
            ("3", "different text", 0),
        ]
    )
    test_frame = create_labeled_frame([])

    cleaned_frame, statistics = clean_training_pool(
        training_frame=training_frame,
        test_frame=test_frame,
        text_column="text",
        label_column="label",
    )

    assert cleaned_frame["id"].tolist() == ["3"]
    assert statistics["removed_ambiguous_rows"] == 2
    assert statistics["final_training_pool_rows"] == 1