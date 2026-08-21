"""
train.py

Trains and compares multiple classification models for predicting a
student's `performance_category` (Low / Medium / High).

Models selected
----------------
* Logistic Regression   - fast, interpretable linear baseline.
* Decision Tree          - simple non-linear baseline, easy to explain.
* Random Forest          - stronger ensemble, resistant to overfitting.
* Gradient Boosting      - typically the strongest tabular-data model
                            among these; used to see if extra
                            complexity is actually justified here.

These four span "simple baseline" to "state-of-the-practice ensemble"
without adding models that would not realistically be compared in a
tabular classification study.

Run directly to execute the full training + evaluation pipeline:

    python -m src.train
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import config, visualization
from src.data_loader import load_raw_dataset
from src.evaluate import evaluate_classifier
from src.preprocessing import build_preprocessing_pipeline, split_features_target


@dataclass
class TrainedModelResult:
    name: str
    pipeline: Pipeline
    cv_mean_f1: float
    cv_std_f1: float
    test_metrics: dict


def get_candidate_models() -> dict[str, object]:
    """Return the estimator instances compared during training."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=config.RANDOM_SEED
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, random_state=config.RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=config.RANDOM_SEED, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=config.RANDOM_SEED),
    }


def train_and_compare_models() -> tuple[TrainedModelResult, list[TrainedModelResult]]:
    """Train every candidate model, evaluate it, and pick the best one.

    Selection is based on mean cross-validated F1 (macro-averaged) on
    the training split, then confirmed against the held-out test set.
    A model is never chosen solely because of its training-set score.
    """
    config.ensure_directories()

    df = load_raw_dataset()
    x, y = split_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )

    cv = StratifiedKFold(
        n_splits=config.CROSS_VALIDATION_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )

    results: list[TrainedModelResult] = []

    for name, estimator in get_candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_preprocessing_pipeline()),
                ("model", estimator),
            ]
        )

        cv_scores = cross_val_score(
            pipeline, x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        test_metrics = evaluate_classifier(y_test, y_pred, labels=config.PERFORMANCE_CATEGORIES)

        results.append(
            TrainedModelResult(
                name=name,
                pipeline=pipeline,
                cv_mean_f1=float(np.mean(cv_scores)),
                cv_std_f1=float(np.std(cv_scores)),
                test_metrics=test_metrics,
            )
        )
        print(
            f"[{name}] CV macro-F1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f}) "
            f"| Test macro-F1: {test_metrics['f1_macro']:.4f}"
        )

    best_result = max(results, key=lambda r: r.test_metrics["f1_macro"])
    print(f"\nBest model selected: {best_result.name} "
          f"(test macro-F1 = {best_result.test_metrics['f1_macro']:.4f})")

    _persist_artifacts(best_result, results, x_test, y_test)
    return best_result, results


def _persist_artifacts(
    best_result: TrainedModelResult,
    all_results: list[TrainedModelResult],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Save the best model, comparison metrics, and diagnostic plots."""
    joblib.dump(best_result.pipeline, config.BEST_MODEL_PATH)

    metadata = {
        "best_model_name": best_result.name,
        "target_column": config.TARGET_COLUMN,
        "performance_categories": config.PERFORMANCE_CATEGORIES,
        "numerical_features": config.NUMERICAL_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "random_seed": config.RANDOM_SEED,
        "test_size": config.TEST_SIZE,
    }
    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    comparison = {
        result.name: {
            "cv_mean_f1_macro": result.cv_mean_f1,
            "cv_std_f1_macro": result.cv_std_f1,
            "test_metrics": result.test_metrics,
        }
        for result in all_results
    }
    config.METRICS_REPORT_PATH.write_text(json.dumps(comparison, indent=2))
    config.BEST_MODEL_REPORT_PATH.write_text(
        json.dumps(
            {"best_model_name": best_result.name, "test_metrics": best_result.test_metrics},
            indent=2,
        )
    )

    # Diagnostic plots for the best model.
    y_pred = best_result.pipeline.predict(x_test)
    visualization.plot_confusion_matrix(y_test, y_pred, labels=config.PERFORMANCE_CATEGORIES)

    comparison_scores = {name: r.test_metrics["f1_macro"] for name, r in
                          zip([r.name for r in all_results], all_results)}
    visualization.plot_model_comparison(comparison_scores, metric_name="Test Macro F1-Score")

    fitted_model = best_result.pipeline.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        preprocessor = best_result.pipeline.named_steps["preprocessing"]
        feature_names = preprocessor.get_feature_names_out()
        importances = pd.Series(fitted_model.feature_importances_, index=feature_names)
        visualization.plot_feature_importance(importances)


def main() -> None:
    train_and_compare_models()


if __name__ == "__main__":
    main()
