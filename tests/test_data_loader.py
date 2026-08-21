"""Unit tests for src/data_loader.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import config
from src.data_loader import (
    DatasetNotFoundError,
    DatasetSchemaError,
    get_feature_target_columns,
    load_dataset,
)


class TestLoadDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_missing_file_raises_dataset_not_found_error(self) -> None:
        missing_path = self.tmp_path / "does_not_exist.csv"
        with self.assertRaises(DatasetNotFoundError):
            load_dataset(missing_path)

    def test_missing_required_columns_raises_schema_error(self) -> None:
        incomplete_csv = self.tmp_path / "incomplete.csv"
        pd.DataFrame({"student_id": [1, 2], "study_hours_per_week": [10, 12]}).to_csv(
            incomplete_csv, index=False
        )
        with self.assertRaises(DatasetSchemaError):
            load_dataset(incomplete_csv, expected_columns=config.EXPECTED_COLUMNS)

    def test_empty_file_raises_value_error(self) -> None:
        empty_csv = self.tmp_path / "empty.csv"
        empty_csv.write_text("")
        with self.assertRaises(ValueError):
            load_dataset(empty_csv, expected_columns=[])

    def test_valid_file_loads_successfully(self) -> None:
        valid_csv = self.tmp_path / "valid.csv"
        df = pd.DataFrame({column: [0] for column in config.EXPECTED_COLUMNS})
        df.to_csv(valid_csv, index=False)

        loaded = load_dataset(valid_csv, expected_columns=config.EXPECTED_COLUMNS)
        self.assertEqual(len(loaded), 1)
        self.assertListEqual(sorted(loaded.columns), sorted(config.EXPECTED_COLUMNS))

    def test_get_feature_target_columns_excludes_non_feature_columns(self) -> None:
        df = pd.DataFrame({column: [0] for column in config.EXPECTED_COLUMNS})
        feature_columns, target_column = get_feature_target_columns(df)

        self.assertNotIn("student_id", feature_columns)
        self.assertNotIn("final_score", feature_columns)
        self.assertNotIn(config.TARGET_COLUMN, feature_columns)
        self.assertEqual(target_column, config.TARGET_COLUMN)


if __name__ == "__main__":
    unittest.main()
