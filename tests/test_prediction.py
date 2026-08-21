"""Unit tests for src/predict.py."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src import config
from src.predict import (
    InvalidStudentInputError,
    ModelNotTrainedError,
    load_trained_pipeline,
    predict_student_performance,
    validate_student_input,
)

VALID_STUDENT = {
    "study_hours_per_week": 20,
    "attendance_rate": 90,
    "previous_grade_avg": 75,
    "sleep_hours_per_night": 7,
    "socioeconomic_index": 55,
    "midterm_score": 78,
    "parental_support_level": "High",
    "extracurricular_activities": "Regular",
    "part_time_job": "No",
    "internet_access_quality": "Good",
}


class TestValidateStudentInput(unittest.TestCase):
    def test_valid_input_passes(self) -> None:
        validated = validate_student_input(VALID_STUDENT)
        self.assertEqual(validated["study_hours_per_week"], 20.0)

    def test_missing_field_raises(self) -> None:
        incomplete = dict(VALID_STUDENT)
        del incomplete["midterm_score"]
        with self.assertRaises(InvalidStudentInputError):
            validate_student_input(incomplete)

    def test_non_numeric_value_raises(self) -> None:
        bad_input = dict(VALID_STUDENT)
        bad_input["attendance_rate"] = "not-a-number"
        with self.assertRaises(InvalidStudentInputError):
            validate_student_input(bad_input)

    def test_out_of_range_value_raises(self) -> None:
        bad_input = dict(VALID_STUDENT)
        bad_input["attendance_rate"] = 150
        with self.assertRaises(InvalidStudentInputError):
            validate_student_input(bad_input)


class TestLoadTrainedPipeline(unittest.TestCase):
    def test_missing_model_raises_model_not_trained_error(self) -> None:
        with patch.object(config, "BEST_MODEL_PATH", config.MODELS_DIR / "does_not_exist.joblib"):
            with self.assertRaises(ModelNotTrainedError):
                load_trained_pipeline()


class TestPredictStudentPerformance(unittest.TestCase):
    def test_prediction_returns_valid_category_if_model_exists(self) -> None:
        if not config.BEST_MODEL_PATH.exists():
            self.skipTest("No trained model artifact present; run `python -m src.train` first.")
        result = predict_student_performance(VALID_STUDENT)
        self.assertIn(result["predicted_category"], config.PERFORMANCE_CATEGORIES)

    def test_prediction_rejects_invalid_input_before_touching_model(self) -> None:
        bad_input = dict(VALID_STUDENT)
        bad_input["sleep_hours_per_night"] = -5
        with self.assertRaises(InvalidStudentInputError):
            predict_student_performance(bad_input)


if __name__ == "__main__":
    unittest.main()
