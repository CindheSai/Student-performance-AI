"""
data_loader.py

Robust loading and validation of the student-performance dataset.

This module intentionally fails loudly rather than silently on
serious data problems (missing file, missing required columns), since
hiding those problems would let broken data flow silently into model
training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the requested dataset file does not exist on disk."""


class DatasetSchemaError(ValueError):
    """Raised when a loaded dataset is missing one or more required columns."""


def load_dataset(
    path: Path = config.RAW_DATASET_PATH,
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a CSV dataset and validate that it matches the expected schema.

    Parameters
    ----------
    path:
        Path to the CSV file to load. Defaults to the project's raw
        dataset path.
    expected_columns:
        Column names that must be present in the loaded dataframe.
        Defaults to `config.EXPECTED_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.

    Raises
    ------
    DatasetNotFoundError
        If `path` does not point to an existing file.
    DatasetSchemaError
        If the loaded dataframe is missing required columns.
    ValueError
        If the file exists but cannot be parsed as CSV, or is empty.
    """
    if expected_columns is None:
        expected_columns = config.EXPECTED_COLUMNS

    if not path.exists():
        raise DatasetNotFoundError(
            f"Dataset file not found at '{path}'. "
            "Run `python -m src.generate_dataset` to create the synthetic "
            "dataset, or place a compatible CSV at this path."
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file at '{path}' is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Dataset file at '{path}' could not be parsed as CSV.") from exc

    if df.empty:
        raise ValueError(f"Dataset loaded from '{path}' contains zero rows.")

    missing_columns = sorted(set(expected_columns) - set(df.columns))
    if missing_columns:
        raise DatasetSchemaError(
            "Dataset is missing required column(s): "
            f"{missing_columns}. Found columns: {sorted(df.columns)}"
        )

    return df


def load_raw_dataset() -> pd.DataFrame:
    """Convenience wrapper to load the project's raw dataset."""
    return load_dataset(config.RAW_DATASET_PATH, config.EXPECTED_COLUMNS)


def get_feature_target_columns(df: pd.DataFrame) -> tuple[list[str], str]:
    """Determine the feature columns and target column for modelling.

    Excludes identifier/leakage columns defined in
    `config.NON_FEATURE_COLUMNS`.
    """
    feature_columns = [
        column for column in df.columns if column not in config.NON_FEATURE_COLUMNS
    ]
    return feature_columns, config.TARGET_COLUMN
