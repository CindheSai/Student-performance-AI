"""Unit tests for src/preprocessing.py."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import build_preprocessing_pipeline, split_features_target


def _make_sample_dataframe(n_rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "student_id": np.arange(n_rows),
            "study_hours_per_week": rng.uniform(0, 30, n_rows),
            "attendance_rate": rng.uniform(50, 100, n_rows),
            "previous_grade_avg": rng.uniform(40, 100, n_rows),
            "sleep_hours_per_night": rng.uniform(4, 9, n_rows),
            "parental_support_level": rng.choice(["Low", "Medium", "High"], n_rows),
            "extracurricular_activities": rng.choice(["None", "Occasional", "Regular"], n_rows),
            "part_time_job": rng.choice(["Yes", "No"], n_rows),
            "internet_access_quality": rng.choice(["Poor", "Average", "Good"], n_rows),
            "socioeconomic_index": rng.uniform(0, 100, n_rows),
            "midterm_score": rng.uniform(0, 100, n_rows),
            "final_score": rng.uniform(0, 100, n_rows),
            "performance_category": rng.choice(["Low", "Medium", "High"], n_rows),
        }
    )
    # Inject missing values to exercise the imputers.
    df.loc[0, "sleep_hours_per_night"] = np.nan
    df.loc[1, "internet_access_quality"] = np.nan
    return df


class TestSplitFeaturesTarget(unittest.TestCase):
    def test_drops_non_feature_columns(self) -> None:
        df = _make_sample_dataframe()
        x, y = split_features_target(df)

        for column in config.NON_FEATURE_COLUMNS:
            self.assertNotIn(column, x.columns)
        self.assertEqual(len(y), len(df))
        self.assertTrue(set(y.unique()).issubset(set(config.PERFORMANCE_CATEGORIES)))


class TestPreprocessingPipeline(unittest.TestCase):
    def test_pipeline_fits_and_transforms_with_missing_values(self) -> None:
        df = _make_sample_dataframe()
        x, _ = split_features_target(df)

        preprocessor = build_preprocessing_pipeline()
        transformed = preprocessor.fit_transform(x)

        # No NaNs should remain after imputation.
        self.assertFalse(np.isnan(transformed).any())
        # Output should have more columns than input due to one-hot encoding.
        self.assertGreater(transformed.shape[1], x.shape[1])
        self.assertEqual(transformed.shape[0], len(df))

    def test_pipeline_is_unfitted_until_explicitly_fit(self) -> None:
        preprocessor = build_preprocessing_pipeline()
        with self.assertRaises(Exception):
            # Calling transform before fit must fail (sklearn's NotFittedError).
            preprocessor.transform(_make_sample_dataframe())

    def test_fit_on_train_only_prevents_leakage_of_unseen_categories(self) -> None:
        df = _make_sample_dataframe(n_rows=30)
        x, _ = split_features_target(df)
        train_x = x.iloc[:20]
        test_x = x.iloc[20:].copy()
        # Introduce a category unseen during training.
        test_x.loc[test_x.index[0], "parental_support_level"] = "Unseen_Category"

        preprocessor = build_preprocessing_pipeline()
        preprocessor.fit(train_x)
        # Must not raise, thanks to handle_unknown="ignore".
        transformed = preprocessor.transform(test_x)
        self.assertEqual(transformed.shape[0], len(test_x))


if __name__ == "__main__":
    unittest.main()
