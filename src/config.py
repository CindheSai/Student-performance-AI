"""
config.py

Centralized configuration for the Student Performance AI System.

All file-system paths, model parameters, and shared constants are
defined here so that no other module hard-codes a path. Using
`pathlib.Path` keeps everything portable across Windows, macOS, and
Linux.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"

# ---------------------------------------------------------------------------
# Dataset files
# ---------------------------------------------------------------------------
RAW_DATASET_PATH: Path = RAW_DATA_DIR / "student_performance_raw.csv"
PROCESSED_DATASET_PATH: Path = PROCESSED_DATA_DIR / "student_performance_processed.csv"

# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Output artifacts
# ---------------------------------------------------------------------------
METRICS_REPORT_PATH: Path = METRICS_DIR / "model_comparison.json"
BEST_MODEL_REPORT_PATH: Path = METRICS_DIR / "best_model_report.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# ML task definition
# ---------------------------------------------------------------------------
# The target is a categorical "Performance Category" derived from a
# student's final academic score. See preprocessing.py /
# data generation notes in README.md for the exact derivation rule.
TARGET_COLUMN: str = "performance_category"
TEST_SIZE: float = 0.2
CROSS_VALIDATION_FOLDS: int = 5

# Columns expected in the raw dataset. Used by data_loader.py to
# validate that a CSV matches the schema this project expects.
EXPECTED_COLUMNS: list[str] = [
    "student_id",
    "study_hours_per_week",
    "attendance_rate",
    "previous_grade_avg",
    "sleep_hours_per_night",
    "parental_support_level",
    "extracurricular_activities",
    "part_time_job",
    "internet_access_quality",
    "socioeconomic_index",
    "midterm_score",
    "final_score",
    "performance_category",
]

NUMERICAL_FEATURES: list[str] = [
    "study_hours_per_week",
    "attendance_rate",
    "previous_grade_avg",
    "sleep_hours_per_night",
    "socioeconomic_index",
    "midterm_score",
]

CATEGORICAL_FEATURES: list[str] = [
    "parental_support_level",
    "extracurricular_activities",
    "part_time_job",
    "internet_access_quality",
]

# Columns that must never be used as model features because they are
# identifiers or would leak the target (final_score directly determines
# performance_category, so it is excluded from the feature set).
NON_FEATURE_COLUMNS: list[str] = ["student_id", "final_score", TARGET_COLUMN]

PERFORMANCE_CATEGORIES: list[str] = ["Low", "Medium", "High"]


def ensure_directories() -> None:
    """Create every project directory this config references, if missing."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
