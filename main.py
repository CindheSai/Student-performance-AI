"""
main.py

Command-line entry point for the Student Performance AI System.
Provides a simple, beginner-friendly menu that ties together dataset
analysis, training, evaluation, and prediction.

Run with:

    python main.py
"""

from __future__ import annotations

import sys

from src import config, visualization
from src.data_loader import DatasetNotFoundError, DatasetSchemaError, load_raw_dataset
from src.evaluate import summarize_metrics
from src.predict import (
    InvalidStudentInputError,
    ModelNotTrainedError,
    predict_student_performance,
    save_prediction_log,
)
from src.train import train_and_compare_models

MENU_TEXT = """
========================================
 STUDENT PERFORMANCE AI SYSTEM
========================================

1. Analyze Dataset
2. Train Models
3. Evaluate Models (show best-model report)
4. Make Prediction
5. Exit
"""


def analyze_dataset() -> None:
    try:
        df = load_raw_dataset()
    except (DatasetNotFoundError, DatasetSchemaError) as exc:
        print(f"\n[Error] {exc}")
        print("Tip: run `python -m src.generate_dataset` to create the dataset.\n")
        return

    print(f"\nLoaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    print("\nClass balance:")
    print(df[config.TARGET_COLUMN].value_counts())

    print("\nGenerating exploratory figures in outputs/figures/ ...")
    visualization.plot_target_distribution(df)
    visualization.plot_numerical_distributions(df)
    visualization.plot_correlation_heatmap(df)
    visualization.plot_categorical_analysis(df)
    print("Done. See outputs/figures/ for saved plots.\n")


def train_models() -> None:
    print("\nTraining models. This may take a moment...\n")
    try:
        best_result, all_results = train_and_compare_models()
    except (DatasetNotFoundError, DatasetSchemaError) as exc:
        print(f"\n[Error] {exc}\n")
        return

    print(f"\nBest model: {best_result.name}")
    print(summarize_metrics(best_result.test_metrics))
    print(f"\nModel and reports saved under: {config.MODELS_DIR} and {config.METRICS_DIR}\n")


def evaluate_models() -> None:
    if not config.BEST_MODEL_REPORT_PATH.exists():
        print("\nNo evaluation report found yet. Train models first (option 2).\n")
        return
    import json

    report = json.loads(config.BEST_MODEL_REPORT_PATH.read_text())
    print(f"\nBest model: {report['best_model_name']}")
    print(summarize_metrics(report["test_metrics"]))
    print()


def make_prediction() -> None:
    print("\nEnter student details (numeric fields) or press Ctrl+C to cancel.\n")
    try:
        student = {
            "study_hours_per_week": input("Study hours per week (0-80): "),
            "attendance_rate": input("Attendance rate % (0-100): "),
            "previous_grade_avg": input("Previous grade average (0-100): "),
            "sleep_hours_per_night": input("Sleep hours per night (0-14): "),
            "socioeconomic_index": input("Socioeconomic index (0-100): "),
            "midterm_score": input("Midterm score (0-100): "),
            "parental_support_level": input("Parental support level (Low/Medium/High): "),
            "extracurricular_activities": input(
                "Extracurricular activities (None/Occasional/Regular): "
            ),
            "part_time_job": input("Part-time job (Yes/No): "),
            "internet_access_quality": input("Internet access quality (Poor/Average/Good): "),
        }
    except KeyboardInterrupt:
        print("\nCancelled.\n")
        return

    try:
        result = predict_student_performance(student)
    except ModelNotTrainedError as exc:
        print(f"\n[Error] {exc}\n")
        return
    except InvalidStudentInputError as exc:
        print(f"\n[Invalid input] {exc}\n")
        return

    save_prediction_log(result)
    print(f"\nPredicted performance category: {result['predicted_category']}")
    if result["class_probabilities"]:
        print("Class probabilities:")
        for category, probability in result["class_probabilities"].items():
            print(f"  {category}: {probability:.3f}")
    print()


def run() -> None:
    config.ensure_directories()
    actions = {
        "1": analyze_dataset,
        "2": train_models,
        "3": evaluate_models,
        "4": make_prediction,
    }

    while True:
        print(MENU_TEXT)
        choice = input("Select an option (1-5): ").strip()

        if choice == "5":
            print("Goodbye.")
            sys.exit(0)

        action = actions.get(choice)
        if action is None:
            print("\nInvalid option. Please choose a number between 1 and 5.\n")
            continue

        action()


if __name__ == "__main__":
    run()
