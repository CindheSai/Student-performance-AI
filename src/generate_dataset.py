"""
generate_dataset.py

Generates the synthetic raw dataset used by this project.

WHY SYNTHETIC DATA
-------------------
This project targets a general "student performance" scenario rather
than one specific institution. Public datasets that match this scope
either carry redistribution restrictions or cannot be reliably fetched
in every environment this project might run in. To keep the project
100% reproducible and legally unencumbered, a synthetic dataset is
generated locally using documented, statistically reasonable rules.

This is NOT real student data and must never be interpreted as
representing an actual student population. See README.md, section
"Dataset", for full disclosure.

GENERATION LOGIC
-----------------
Each synthetic student is described by behavioural and academic
features (study hours, attendance, prior grades, sleep, support,
etc.). A continuous `final_score` is generated as a weighted, noisy
combination of those features, deliberately mirroring realistic
education-research findings (e.g. attendance and prior grades matter
more than any single behavioural factor). The categorical target,
`performance_category`, is then derived directly from `final_score`
using fixed thresholds -- so the label is fully traceable and free of
hidden/artificial logic.

Run directly to (re)write data/raw/student_performance_raw.csv:

    python -m src.generate_dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def _clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip an array into a plausible real-world range."""
    return np.clip(values, low, high)


def generate_synthetic_dataset(
    n_students: int = 1200, random_seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """Generate a synthetic student-performance dataset.

    Parameters
    ----------
    n_students:
        Number of synthetic student records to generate.
    random_seed:
        Seed for NumPy's random generator, so the dataset is
        reproducible across runs.

    Returns
    -------
    pandas.DataFrame
        A dataframe with the schema described in
        `config.EXPECTED_COLUMNS`.
    """
    rng = np.random.default_rng(random_seed)

    student_id = np.arange(1, n_students + 1)

    study_hours_per_week = _clip(rng.normal(loc=15, scale=6, size=n_students), 0, 40)
    attendance_rate = _clip(rng.normal(loc=82, scale=12, size=n_students), 30, 100)
    previous_grade_avg = _clip(rng.normal(loc=68, scale=14, size=n_students), 20, 100)
    sleep_hours_per_night = _clip(rng.normal(loc=6.8, scale=1.3, size=n_students), 3, 10)
    socioeconomic_index = _clip(rng.normal(loc=50, scale=20, size=n_students), 0, 100)

    parental_support_level = rng.choice(
        ["Low", "Medium", "High"], size=n_students, p=[0.25, 0.45, 0.30]
    )
    extracurricular_activities = rng.choice(
        ["None", "Occasional", "Regular"], size=n_students, p=[0.30, 0.40, 0.30]
    )
    part_time_job = rng.choice(["Yes", "No"], size=n_students, p=[0.35, 0.65])
    internet_access_quality = rng.choice(
        ["Poor", "Average", "Good"], size=n_students, p=[0.15, 0.40, 0.45]
    )

    # Midterm score correlates with prior grades and study habits, plus noise.
    midterm_score = _clip(
        0.55 * previous_grade_avg
        + 0.9 * study_hours_per_week
        + rng.normal(0, 8, size=n_students),
        0,
        100,
    )

    # Map categorical effects to numeric adjustments for the final score.
    support_bonus = pd.Series(parental_support_level).map(
        {"Low": -3.0, "Medium": 0.0, "High": 3.5}
    ).to_numpy()
    activity_bonus = pd.Series(extracurricular_activities).map(
        {"None": -1.0, "Occasional": 1.0, "Regular": 2.0}
    ).to_numpy()
    job_penalty = pd.Series(part_time_job).map({"Yes": -4.0, "No": 0.0}).to_numpy()
    internet_bonus = pd.Series(internet_access_quality).map(
        {"Poor": -3.5, "Average": 0.0, "Good": 3.0}
    ).to_numpy()

    # Final score: weighted combination of academic, behavioural, and
    # environmental factors plus Gaussian noise. Weights are chosen to
    # be directionally realistic, not derived from any real dataset.
    final_score = (
        0.30 * previous_grade_avg
        + 0.25 * midterm_score
        + 0.9 * study_hours_per_week
        + 0.15 * attendance_rate
        + 0.4 * sleep_hours_per_night
        + 0.05 * socioeconomic_index
        + support_bonus
        + activity_bonus
        + job_penalty
        + internet_bonus
        + rng.normal(0, 6, size=n_students)
    )
    final_score = _clip(final_score, 0, 100)

    def categorize(score: float) -> str:
        if score < 55:
            return "Low"
        if score < 75:
            return "Medium"
        return "High"

    performance_category = np.array([categorize(s) for s in final_score])

    df = pd.DataFrame(
        {
            "student_id": student_id,
            "study_hours_per_week": np.round(study_hours_per_week, 2),
            "attendance_rate": np.round(attendance_rate, 2),
            "previous_grade_avg": np.round(previous_grade_avg, 2),
            "sleep_hours_per_night": np.round(sleep_hours_per_night, 2),
            "parental_support_level": parental_support_level,
            "extracurricular_activities": extracurricular_activities,
            "part_time_job": part_time_job,
            "internet_access_quality": internet_access_quality,
            "socioeconomic_index": np.round(socioeconomic_index, 2),
            "midterm_score": np.round(midterm_score, 2),
            "final_score": np.round(final_score, 2),
            "performance_category": performance_category,
        }
    )

    # Introduce a small, realistic amount of missing data so that
    # preprocessing.py's missing-value handling is exercised meaningfully.
    missing_rng = np.random.default_rng(random_seed + 1)
    for column in ["sleep_hours_per_night", "internet_access_quality", "attendance_rate"]:
        missing_idx = missing_rng.choice(
            n_students, size=int(n_students * 0.02), replace=False
        )
        df.loc[missing_idx, column] = np.nan

    return df


def main() -> None:
    """Generate the dataset and write it to `config.RAW_DATASET_PATH`."""
    config.ensure_directories()
    df = generate_synthetic_dataset()
    df.to_csv(config.RAW_DATASET_PATH, index=False)
    print(f"Synthetic dataset written to: {config.RAW_DATASET_PATH}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(df["performance_category"].value_counts())


if __name__ == "__main__":
    main()
