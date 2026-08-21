"""
visualization.py

Reusable, purpose-driven plotting functions for the student-performance
project. Every function saves a figure into `outputs/figures/` and
answers a specific analytical question rather than existing merely to
pad the project with extra files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for script/CI execution

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src import config

plt.rcParams["figure.autolayout"] = True


def _save(fig: plt.Figure, filename: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_target_distribution(df: pd.DataFrame, filename: str = "target_distribution.png") -> Path:
    """Answers: How balanced are the three performance categories?"""
    counts = df[config.TARGET_COLUMN].value_counts().reindex(config.PERFORMANCE_CATEGORIES)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=["#d9534f", "#f0ad4e", "#5cb85c"])
    ax.set_title("Distribution of Performance Categories")
    ax.set_xlabel("Performance Category")
    ax.set_ylabel("Number of Students")
    for i, value in enumerate(counts.values):
        ax.text(i, value + max(counts.values) * 0.01, str(int(value)), ha="center")
    return _save(fig, filename)


def plot_numerical_distributions(
    df: pd.DataFrame, filename: str = "numerical_feature_distributions.png"
) -> Path:
    """Answers: What does the spread of each numeric feature look like?"""
    features = config.NUMERICAL_FEATURES
    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, feature in zip(axes, features):
        ax.hist(df[feature].dropna(), bins=25, color="#5b8def", edgecolor="white")
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle("Numerical Feature Distributions", fontsize=14)
    return _save(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame, filename: str = "correlation_heatmap.png") -> Path:
    """Answers: Which numeric features move together, and how strongly
    does each relate to the final score?"""
    columns = config.NUMERICAL_FEATURES + ["final_score"]
    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(
                j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                color="black", fontsize=8,
            )
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Correlation: Numerical Features vs Final Score", fontsize=11)
    return _save(fig, filename)


def plot_categorical_analysis(
    df: pd.DataFrame, filename: str = "categorical_feature_analysis.png"
) -> Path:
    """Answers: How does average final score vary across each categorical
    feature's groups?"""
    features = config.CATEGORICAL_FEATURES
    n_cols = 2
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, feature in zip(axes, features):
        grouped = df.groupby(feature)["final_score"].mean().sort_values()
        ax.barh(grouped.index.astype(str), grouped.values, color="#8e7cc3")
        ax.set_title(f"Avg Final Score by {feature}")
        ax.set_xlabel("Average Final Score")

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle("Categorical Feature Analysis", fontsize=14)
    return _save(fig, filename)


def plot_model_comparison(
    results: dict[str, float], metric_name: str, filename: str = "model_comparison.png"
) -> Path:
    """Answers: Which trained model performs best on the chosen metric?"""
    names = list(results.keys())
    values = list(results.values())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, values, color="#5cb85c")
    best_idx = int(np.argmax(values))
    bars[best_idx].set_color("#d9534f")
    ax.set_title(f"Model Comparison ({metric_name})")
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, 1)
    for i, value in enumerate(values):
        ax.text(i, value + 0.01, f"{value:.3f}", ha="center")
    plt.xticks(rotation=15, ha="right")
    return _save(fig, filename)


def plot_confusion_matrix(
    y_true, y_pred, labels: list[str], filename: str = "confusion_matrix_best_model.png"
) -> Path:
    """Answers: Where does the best model's predictions get confused
    between categories?"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, cmap="Blues", ax=ax, colorbar=True
    )
    ax.set_title("Confusion Matrix - Best Model")
    return _save(fig, filename)


def plot_feature_importance(
    importances: pd.Series, filename: str = "feature_importance.png", top_n: int = 15
) -> Path:
    """Answers: Which engineered features drive the best model's
    predictions the most? (Only meaningful for models that expose
    feature importances.)"""
    top = importances.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(top))))
    ax.barh(top.index, top.values, color="#f0ad4e")
    ax.set_title("Top Feature Importances - Best Model")
    ax.set_xlabel("Importance")
    return _save(fig, filename)
