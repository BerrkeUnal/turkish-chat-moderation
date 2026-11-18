from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from chat_moderation.config import resolve_project_path
from chat_moderation.preprocessing import canonicalize_text


@dataclass(frozen=True)
class PreparedDataPaths:
    """Paths created by the data preparation pipeline."""

    train: Path
    validation: Path
    test: Path
    statistics: Path


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_archive(url: str, destination: Path, force: bool = False) -> Path:
    """Download an archive atomically."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    partial_path = destination.with_suffix(f"{destination.suffix}.part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "turkish-chat-moderation/0.1.0"},
    )

    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            partial_path.open("wb") as output_file,
        ):
            shutil.copyfileobj(response, output_file)

        if partial_path.stat().st_size == 0:
            raise RuntimeError("Downloaded archive is empty.")

        partial_path.replace(destination)
    finally:
        partial_path.unlink(missing_ok=True)

    return destination


def extract_required_files(
    archive_path: Path,
    raw_data_dir: Path,
    data_config: Mapping[str, Any],
) -> dict[str, Path]:
    """Extract only the required dataset files."""

    raw_data_dir.mkdir(parents=True, exist_ok=True)

    required_members = {
        "train": (
            data_config["train_member"],
            raw_data_dir / "training.tsv",
        ),
        "test": (
            data_config["test_member"],
            raw_data_dir / "test.tsv",
        ),
        "test_labels": (
            data_config["test_labels_member"],
            raw_data_dir / "test_labels.csv",
        ),
    }

    extracted_paths: dict[str, Path] = {}

    with zipfile.ZipFile(archive_path) as archive:
        available_members = set(archive.namelist())

        for key, (member_name, destination) in required_members.items():
            if member_name not in available_members:
                raise FileNotFoundError(
                    f"Required file is missing from the archive: {member_name}"
                )

            partial_path = destination.with_suffix(f"{destination.suffix}.part")

            try:
                with (
                    archive.open(member_name) as source_file,
                    partial_path.open("wb") as output_file,
                ):
                    shutil.copyfileobj(source_file, output_file)

                partial_path.replace(destination)
            finally:
                partial_path.unlink(missing_ok=True)

            extracted_paths[key] = destination

    return extracted_paths


def validate_split_ids(
    splits: Mapping[str, pd.DataFrame],
    id_column: str = "id",
) -> None:
    """Ensure that dataset splits have no overlapping identifiers."""

    for left_name, right_name in combinations(splits, 2):
        left_ids = set(splits[left_name][id_column].astype(str))
        right_ids = set(splits[right_name][id_column].astype(str))
        overlap = left_ids.intersection(right_ids)

        if overlap:
            raise ValueError(
                f"ID overlap between {left_name} and {right_name}: "
                f"{len(overlap)} identifiers"
            )

def canonicalize_text_for_split(text: str) -> str:
    """Create a stable text key for duplicate and leakage checks."""

    return canonicalize_text(text)

def clean_training_pool(
    training_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove ambiguous, duplicate and test-overlapping training texts."""

    canonical_column = "_canonical_text"
    cleaned_frame = training_frame.copy()

    cleaned_frame[canonical_column] = cleaned_frame[text_column].map(
        canonicalize_text_for_split
    )

    label_counts = cleaned_frame.groupby(canonical_column)[
        label_column
    ].nunique()

    ambiguous_keys = set(label_counts[label_counts > 1].index)
    ambiguous_mask = cleaned_frame[canonical_column].isin(ambiguous_keys)
    removed_ambiguous_rows = int(ambiguous_mask.sum())

    cleaned_frame = cleaned_frame.loc[~ambiguous_mask].copy()

    rows_before_deduplication = len(cleaned_frame)
    cleaned_frame = cleaned_frame.drop_duplicates(
        subset=canonical_column,
        keep="first",
    )
    removed_duplicate_rows = rows_before_deduplication - len(cleaned_frame)

    test_keys = set(
        test_frame[text_column].map(canonicalize_text_for_split)
    )
    test_overlap_mask = cleaned_frame[canonical_column].isin(test_keys)
    removed_test_overlap_rows = int(test_overlap_mask.sum())

    cleaned_frame = cleaned_frame.loc[~test_overlap_mask].copy()
    cleaned_frame = cleaned_frame.drop(columns=canonical_column)

    statistics = {
        "removed_ambiguous_rows": removed_ambiguous_rows,
        "removed_duplicate_rows": int(removed_duplicate_rows),
        "removed_test_overlap_rows": removed_test_overlap_rows,
        "final_training_pool_rows": int(len(cleaned_frame)),
    }


    return cleaned_frame.reset_index(drop=True), statistics

def _standardize_frame(
    frame: pd.DataFrame,
    data_config: Mapping[str, Any],
    label_mapping: Mapping[str, int],
) -> pd.DataFrame:
    id_column = data_config["id_column"]
    text_column = data_config["text_column"]
    label_column = data_config["label_column"]
    raw_text_column = data_config["raw_text_column"]
    raw_label_column = data_config["raw_label_column"]

    required_columns = {id_column, raw_text_column, raw_label_column}
    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        raise ValueError(f"Dataset columns are missing: {sorted(missing_columns)}")

    standardized = frame[
        [id_column, raw_text_column, raw_label_column]
    ].copy()

    standardized = standardized.rename(
        columns={
            raw_text_column: text_column,
            raw_label_column: label_column,
        }
    )

    standardized[id_column] = (
        standardized[id_column].astype("string").str.strip()
    )
    standardized[text_column] = (
        standardized[text_column].astype("string").str.strip()
    )
    standardized[label_column] = (
        standardized[label_column]
        .astype("string")
        .str.strip()
        .map(label_mapping)
    )

    if standardized[label_column].isna().any():
        invalid_labels = sorted(
            frame.loc[
                standardized[label_column].isna(),
                raw_label_column,
            ]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Unknown labels found: {invalid_labels}")

    if standardized[id_column].duplicated().any():
        raise ValueError("Duplicate identifiers were found in the dataset.")

    if standardized[text_column].eq("").any():
        raise ValueError("Empty text values were found in the dataset.")

    standardized[label_column] = standardized[label_column].astype("int64")

    return standardized.reset_index(drop=True)


def load_raw_frames(
    extracted_paths: Mapping[str, Path],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and standardize the official training and test data."""

    data_config = config["data"]
    id_column = data_config["id_column"]
    raw_label_column = data_config["raw_label_column"]

    label_mapping = {
        str(label): int(value)
        for label, value in config["labels"]["raw_mapping"].items()
    }

    training_frame = pd.read_csv(
        extracted_paths["train"],
        sep="\t",
        dtype="string",
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )

    test_frame = pd.read_csv(
        extracted_paths["test"],
        sep="\t",
        dtype="string",
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )

    test_labels = pd.read_csv(
        extracted_paths["test_labels"],
        sep=",",
        names=[id_column, raw_label_column],
        dtype="string",
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )

    test_frame = test_frame.merge(
        test_labels,
        on=id_column,
        how="left",
        validate="one_to_one",
    )

    training_frame = _standardize_frame(
        training_frame,
        data_config,
        label_mapping,
    )
    test_frame = _standardize_frame(
        test_frame,
        data_config,
        label_mapping,
    )

    return training_frame, test_frame


def _create_split_statistics(
    frame: pd.DataFrame,
    label_column: str,
) -> dict[str, int | float]:
    label_counts = frame[label_column].value_counts()

    clean_count = int(label_counts.get(0, 0))
    offensive_count = int(label_counts.get(1, 0))
    row_count = int(len(frame))

    return {
        "rows": row_count,
        "clean": clean_count,
        "offensive": offensive_count,
        "offensive_ratio": round(offensive_count / row_count, 6),
    }

def prepare_dataset(
    config: Mapping[str, Any],
    force_download: bool = False,
) -> tuple[PreparedDataPaths, dict[str, Any]]:
    """Download, validate, clean, split and save the dataset."""

    data_config = config["data"]
    paths_config = config["paths"]

    raw_data_dir = resolve_project_path(paths_config["raw_data_dir"])
    processed_data_dir = resolve_project_path(
        paths_config["processed_data_dir"]
    )

    archive_path = raw_data_dir / data_config["archive_name"]

    download_archive(
        url=data_config["source_url"],
        destination=archive_path,
        force=force_download,
    )

    extracted_paths = extract_required_files(
        archive_path=archive_path,
        raw_data_dir=raw_data_dir,
        data_config=data_config,
    )

    training_frame, test_frame = load_raw_frames(extracted_paths, config)

    label_column = data_config["label_column"]
    id_column = data_config["id_column"]
    text_column = data_config["text_column"]

    source_rows = {
        "training": int(len(training_frame)),
        "test": int(len(test_frame)),
    }

    training_frame, cleaning_statistics = clean_training_pool(
        training_frame=training_frame,
        test_frame=test_frame,
        text_column=text_column,
        label_column=label_column,
    )

    train_frame, validation_frame = train_test_split(
        training_frame,
        test_size=float(data_config["validation_size"]),
        random_state=int(config["project"]["seed"]),
        stratify=training_frame[label_column],
    )

    train_frame = train_frame.reset_index(drop=True)
    validation_frame = validation_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)

    splits = {
        "train": train_frame,
        "validation": validation_frame,
        "test": test_frame,
    }

    validate_split_ids(splits, id_column=id_column)

    processed_data_dir.mkdir(parents=True, exist_ok=True)

    prepared_paths = PreparedDataPaths(
        train=processed_data_dir / "train.csv",
        validation=processed_data_dir / "validation.csv",
        test=processed_data_dir / "test.csv",
        statistics=processed_data_dir / "dataset_statistics.json",
    )

    train_frame.to_csv(prepared_paths.train, index=False)
    validation_frame.to_csv(prepared_paths.validation, index=False)
    test_frame.to_csv(prepared_paths.test, index=False)

    exact_text_sets = {
        split_name: set(frame[text_column].astype(str))
        for split_name, frame in splits.items()
    }

    canonical_text_sets = {
        split_name: set(
            frame[text_column].map(canonicalize_text_for_split)
        )
        for split_name, frame in splits.items()
    }

    statistics: dict[str, Any] = {
        "source_url": data_config["source_url"],
        "archive_sha256": calculate_sha256(archive_path),
        "source_rows": source_rows,
        "cleaning": cleaning_statistics,
        "seed": int(config["project"]["seed"]),
        "validation_size": float(data_config["validation_size"]),
        "splits": {
            split_name: _create_split_statistics(frame, label_column)
            for split_name, frame in splits.items()
        },
        "exact_text_overlap": {
            "train_validation": len(
                exact_text_sets["train"].intersection(
                    exact_text_sets["validation"]
                )
            ),
            "train_test": len(
                exact_text_sets["train"].intersection(
                    exact_text_sets["test"]
                )
            ),
            "validation_test": len(
                exact_text_sets["validation"].intersection(
                    exact_text_sets["test"]
                )
            ),
        },
        "canonical_text_overlap": {
            "train_validation": len(
                canonical_text_sets["train"].intersection(
                    canonical_text_sets["validation"]
                )
            ),
            "train_test": len(
                canonical_text_sets["train"].intersection(
                    canonical_text_sets["test"]
                )
            ),
            "validation_test": len(
                canonical_text_sets["validation"].intersection(
                    canonical_text_sets["test"]
                )
            ),
        },
    }

    with prepared_paths.statistics.open("w", encoding="utf-8") as file:
        json.dump(statistics, file, ensure_ascii=False, indent=2)

    return prepared_paths, statistics