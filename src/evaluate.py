"""
evaluate.py

Classification-appropriate evaluation utilities for the
performance-category models. Only metrics that are mathematically
valid for a multi-class classification task are reported (accuracy,
precision, recall, F1, confusion matrix, classification report) --
regression metrics such as RMSE or R^2 do not apply here and are
intentionally omitted.
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(y_true, y_pred, labels: list[str]) -> dict[str, Any]:
    """Compute a standard set of multi-class classification metrics.

    Macro-averaging is used for precision/recall/F1 so that the
    minority classes (e.g. "High" performers, who are naturally fewer)
    are not drowned out by the majority class.

    Returns
    -------
    dict
        Contains accuracy, macro precision/recall/F1, the confusion
        matrix (as a nested list), and the full per-class
        classification report (as a dict).
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "classification_report": report,
    }


def summarize_metrics(metrics: dict[str, Any]) -> str:
    """Return a short, human-readable summary of a metrics dict."""
    return (
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Precision (macro): {metrics['precision_macro']:.4f} | "
        f"Recall (macro): {metrics['recall_macro']:.4f} | "
        f"F1 (macro): {metrics['f1_macro']:.4f}"
    )
