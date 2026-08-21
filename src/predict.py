"""
predict.py

Clean prediction interface for the trained student-performance model.

The saved artifact at `config.BEST_MODEL_PATH` is a full scikit-learn
`Pipeline` that includes the fitted preprocessing step, so predictions
here reuse exactly the same preprocessing logic that was used during
training -- there is no duplicated transformation code.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd

from src import config


class ModelNotTrainedError(FileNotFoundError):
    """Raised when a prediction is requested before any model has been trained."""


class InvalidStudentInputError(ValueError):
    """Raised when supplied student data fails validation."""


def load_trained_pipeline():
    """Load the persisted best-model pipeline from disk."""
    if not config.BEST_MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"No trained model found at '{config.BEST_MODEL_PATH}'. "
            "Run `python -m src.train` first."
        )
    return joblib.load(config.BEST_MODEL_PATH)


def validate_student_input(student: dict[str, Any]) -> dict[str, Any]:
    """Validate a raw dict of student features before prediction.

    Checks that every required feature is present and that numerical
    fields are actually numeric and within a plausible range.
    """
    required_fields = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
    missing = [field for field in required_fields if field not in student]
    if missing:
        raise InvalidStudentInputError(f"Missing required field(s): {missing}")

    numeric_bounds = {
        "study_hours_per_week": (0, 80),
        "attendance_rate": (0, 100),
        "previous_grade_avg": (0, 100),
        "sleep_hours_per_night": (0, 14),
        "socioeconomic_index": (0, 100),
        "midterm_score": (0, 100),
    }

    validated: dict[str, Any] = dict(student)
    for field, (low, high) in numeric_bounds.items():
        value = validated[field]
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidStudentInputError(
                f"Field '{field}' must be numeric, got: {student[field]!r}"
            ) from exc
        if not (low <= value <= high):
            raise InvalidStudentInputError(
                f"Field '{field}' value {value} is outside the plausible range "
                f"[{low}, {high}]."
            )
        validated[field] = value

    return validated


def predict_student_performance(student: dict[str, Any]) -> dict[str, Any]:
    """Predict a single student's performance category.

    Parameters
    ----------
    student:
        Dict containing every feature listed in
        `config.NUMERICAL_FEATURES` and `config.CATEGORICAL_FEATURES`.

    Returns
    -------
    dict
        `{"predicted_category": str, "class_probabilities": dict | None}`
    """
    validated = validate_student_input(student)
    pipeline = load_trained_pipeline()

    feature_order = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
    input_df = pd.DataFrame([{field: validated[field] for field in feature_order}])

    predicted_category = pipeline.predict(input_df)[0]

    class_probabilities = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_df)[0]
        classes = pipeline.classes_
        class_probabilities = {
            str(cls): float(prob) for cls, prob in zip(classes, probabilities)
        }

    return {
        "predicted_category": str(predicted_category),
        "class_probabilities": class_probabilities,
    }


def predict_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Predict performance categories for every row in a dataframe.

    `df` must contain the same feature columns used during training.
    Returns a copy of `df` with an added `predicted_category` column.
    """
    pipeline = load_trained_pipeline()
    feature_order = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise InvalidStudentInputError(f"Input dataframe missing column(s): {missing}")

    result = df.copy()
    result["predicted_category"] = pipeline.predict(df[feature_order])
    return result


def save_prediction_log(prediction: dict[str, Any], filename: str = "latest_prediction.json") -> None:
    """Persist a single prediction result to `outputs/predictions/`."""
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PREDICTIONS_DIR / filename
    out_path.write_text(json.dumps(prediction, indent=2))
