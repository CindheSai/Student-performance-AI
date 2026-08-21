# Student Performance AI: Prediction & Analysis System

## Overview

This project is an end-to-end machine learning system that predicts a
student's academic **performance category** (`Low`, `Medium`, or
`High`) from academic and behavioural indicators — study habits,
attendance, prior grades, sleep, parental support, and more. It walks
through the complete ML lifecycle: dataset validation, cleaning,
exploratory data analysis, feature engineering, preprocessing, model
training and comparison, evaluation, and a reusable prediction
interface.

It was built as a final AI & ML project to demonstrate applied
proficiency with Python, Pandas, NumPy, Matplotlib, and Scikit-learn
in a realistic, reproducible software structure — not a single
notebook.

## Problem Statement

Educators and academic advisors often want an early, data-informed
signal of which students may need additional support before final
exams. Manually reviewing every student's history at scale isn't
practical. This project frames that need as a **supervised multi-class
classification problem**: given a student's measurable habits and
academic history (excluding their actual final exam outcome), predict
which performance band they are likely to fall into.

## Objectives

- Build a clean, modular, testable ML codebase (not a single script).
- Demonstrate the full pipeline: raw data → validation → cleaning →
  EDA → feature engineering → train/test split → multiple models →
  evaluation → best-model selection → prediction → visualization →
  persisted model.
- Avoid data leakage and inappropriate metrics.
- Produce honest, reproducible results — no fabricated numbers.

## Features

- Synthetic, clearly documented dataset generator (no unlicensed data
  redistribution).
- Schema-validating data loader with descriptive error messages.
- Leakage-safe `ColumnTransformer` preprocessing pipeline (imputation,
  scaling, one-hot encoding) fit only on training data.
- Four compared classification models with cross-validation.
- Full classification metrics (accuracy, macro precision/recall/F1,
  confusion matrix, per-class report).
- Six purpose-built visualizations saved to `outputs/figures/`.
- A prediction interface that reuses the exact training-time
  preprocessing logic (no duplicated transformation code).
- A beginner-friendly CLI (`main.py`).
- 16 passing unit tests covering loading, preprocessing, and
  prediction.

## Machine Learning Approach

**Task type: multi-class classification.**

Performance is predicted as one of three ordinal-but-treated-as-
categorical classes (`Low`, `Medium`, `High`) rather than as a raw
continuous score. This was chosen because:

1. In practice, advisors act on *bands* of risk/performance, not
   precise point estimates — a categorical output is directly
   actionable.
2. It lets the project demonstrate classification-specific techniques
   (macro-averaged precision/recall/F1, confusion matrices,
   stratified cross-validation) that are core ML-engineering skills.

**Avoiding data leakage:** the dataset also contains `final_score`, the
continuous value the category was derived from. `final_score` (and the
non-predictive `student_id`) are explicitly excluded from the feature
set in `config.NON_FEATURE_COLUMNS` and `preprocessing.split_features_target`,
so no model ever sees the value it would otherwise be able to reverse
into a trivial rule.

**Models compared** (see "Results" below for actual scores):

| Model | Why it's included |
|---|---|
| Logistic Regression | Fast, interpretable linear baseline |
| Decision Tree | Simple non-linear baseline, easy to explain |
| Random Forest | Ensemble method, resistant to overfitting |
| Gradient Boosting | Typically the strongest classical tabular-data model |

Model selection is based on **macro-averaged F1 on the held-out test
set**, confirmed against 5-fold stratified cross-validation — never on
training-set accuracy alone.

## Dataset

**This project uses a synthetically generated dataset, not real
student records.**

A reliable, redistribution-safe public dataset matching this exact
project's scope (behavioural + academic features → a categorical
performance label, with a documented derivation) could not be
reliably bundled with the project. To keep the project fully
reproducible and free of licensing concerns, `src/generate_dataset.py`
generates 1,200 synthetic student records using explicit, documented
rules:

- Each numerical feature (study hours, attendance, prior grades,
  sleep, socioeconomic index) is drawn from a bounded normal
  distribution with realistic means/spreads.
- Categorical features (parental support, extracurriculars, part-time
  job, internet access quality) are sampled from fixed probability
  distributions.
- A continuous `final_score` is computed as a weighted combination of
  these features plus Gaussian noise, with weights chosen to be
  directionally realistic (e.g. prior grades and study hours matter
  more than any single categorical factor) — **these weights are not
  derived from any real dataset and should not be read as empirical
  findings.**
- `performance_category` is derived directly from `final_score` using
  fixed thresholds (`< 55` → Low, `< 75` → Medium, else High).
- ~2% missing values are injected into three columns to make the
  missing-value handling in `preprocessing.py` meaningful to test.

**Do not treat this dataset, or any metric derived from it, as
representing a real student population.** Regenerate it at any time
with:

```bash
python -m src.generate_dataset
```

## Project Architecture

```text
student-performance-ai/
│
├── data/
│   ├── raw/                     # generated synthetic CSV lives here
│   └── processed/
│
├── notebooks/
│   └── student_performance_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                # centralized paths & constants
│   ├── generate_dataset.py      # synthetic dataset generator
│   ├── data_loader.py           # validated CSV loading
│   ├── preprocessing.py         # leakage-safe preprocessing pipeline
│   ├── visualization.py         # reusable plotting functions
│   ├── train.py                 # trains & compares 4 models
│   ├── evaluate.py              # classification metrics
│   └── predict.py               # prediction interface
│
├── models/                      # persisted best model + metadata
│
├── outputs/
│   ├── figures/                 # saved PNG plots
│   ├── metrics/                 # saved JSON evaluation reports
│   └── predictions/             # saved prediction logs
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_prediction.py
│
├── .gitignore
├── requirements.txt
├── README.md
├── LICENSE
└── main.py
```

One addition to a typical minimal layout: `src/generate_dataset.py`.
It exists because the project ships a synthetic dataset instead of a
downloaded one (see "Dataset" above), so dataset creation is itself a
pipeline step worth isolating in its own module.

## Technologies Used

- Python 3.10+
- Pandas, NumPy — data handling
- Matplotlib — visualization
- Scikit-learn — preprocessing, models, metrics, pipelines
- Joblib — model persistence
- Jupyter — analysis notebook
- `unittest` — testing

## Installation

```bash
git clone https://github.com/CindheSai/Student-performance-AI.git
cd student-performance-ai
```

## Environment Setup

### Windows

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Generate the dataset (creates `data/raw/student_performance_raw.csv`):

```bash
python -m src.generate_dataset
```

Run the interactive CLI:

```bash
python main.py
```

### Training

```bash
python -m src.train
```

Trains all four candidate models, cross-validates them, evaluates
each on a held-out test set, saves the best pipeline to
`models/best_model.joblib`, and writes comparison plots/metrics to
`outputs/`.

### Evaluation

Evaluation happens automatically during training. To re-inspect the
saved report without retraining:

```bash
python -c "import json; from src import config; print(json.load(open(config.BEST_MODEL_REPORT_PATH)))"
```

### Prediction

From Python:

```python
from src.predict import predict_student_performance

sample = {
    "study_hours_per_week": 22,
    "attendance_rate": 91,
    "previous_grade_avg": 78,
    "sleep_hours_per_night": 7.2,
    "socioeconomic_index": 60,
    "midterm_score": 80,
    "parental_support_level": "High",
    "extracurricular_activities": "Regular",
    "part_time_job": "No",
    "internet_access_quality": "Good",
}
print(predict_student_performance(sample))
```

Or interactively via `python main.py` → option 4.

## Results

**These are actual results from one execution of `python -m src.train`
on the synthetic dataset in this repository (random seed 42, 1,200
rows, 80/20 stratified train/test split). Re-running the pipeline on
a regenerated dataset may produce slightly different numbers.**

| Model | CV macro-F1 (mean ± std) | Test macro-F1 |
|---|---|---|
| Logistic Regression | 0.7362 ± 0.0327 | **0.7368** |
| Decision Tree | 0.5925 ± 0.0124 | 0.5871 |
| Random Forest | 0.6697 ± 0.0178 | 0.6900 |
| Gradient Boosting | 0.6808 ± 0.0227 | 0.6821 |

**Best model: Logistic Regression** (selected by highest test macro-F1,
confirmed by cross-validation).

Best-model test-set metrics:

| Metric | Value |
|---|---|
| Accuracy | 0.7792 |
| Precision (macro) | 0.7804 |
| Recall (macro) | 0.7142 |
| F1 (macro) | 0.7368 |

Per-class breakdown (test set):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 0.6471 | 0.6875 | 0.6667 | 48 |
| Medium | 0.8012 | 0.8600 | 0.8296 | 150 |
| High | 0.8929 | 0.5952 | 0.7143 | 42 |

The linear model outperforming the tree ensembles here is a reasonable
outcome given how the synthetic target was constructed: `final_score`
is a fairly smooth, roughly linear combination of the input features
plus noise, which favours a linear classifier over ensembles tuned for
more complex non-linear interactions. On a dataset with genuinely
non-linear structure, the ranking could easily differ — this result is
specific to this dataset and should not be generalized.

Raw metrics for every model, including full confusion matrices and
per-class reports, are saved at `outputs/metrics/model_comparison.json`.

## Visualizations

All saved to `outputs/figures/`:

- **`target_distribution.png`** — how many students fall into each
  performance category (checks for class imbalance).
- **`numerical_feature_distributions.png`** — histograms of every
  numeric feature, to sanity-check ranges and skew.
- **`correlation_heatmap.png`** — pairwise Pearson correlation between
  numeric features and `final_score`, to see which behaviours track
  most strongly with outcomes.
- **`categorical_feature_analysis.png`** — average final score per
  group within each categorical feature (e.g. does parental support
  level move the average score?).
- **`model_comparison.png`** — bar chart of test macro-F1 across all
  four trained models, best model highlighted.
- **`confusion_matrix_best_model.png`** — where the best model's
  predictions are correct vs. confused between adjacent categories.

## Project Structure

See "Project Architecture" above for the full folder tree.

## Testing

16 unit tests cover dataset loading/validation, the preprocessing
pipeline (including leakage prevention and missing-value handling),
and the prediction interface (including input validation and
model-not-found handling).

```bash
python -m unittest discover -s tests -v
```

All 16 tests pass as of the last run in this environment.

## Limitations

- The dataset is synthetic; no result here should be treated as a
  finding about real students, and the model must not be deployed
  against real student data without re-validation on real,
  ethically-sourced data.
- Class imbalance (Medium students outnumber Low and High combined)
  likely contributes to the "High" class's lower recall — the model
  misses roughly 40% of true High performers on this test set.
- Feature weights used to generate the synthetic target are
  illustrative, not empirically derived, so feature-importance
  interpretations only describe this synthetic dataset's construction,
  not real educational causality.
- No hyperparameter tuning (e.g. grid/random search) was performed;
  default or lightly-adjusted hyperparameters were used throughout.

## Future Improvements

- Replace the synthetic dataset with a properly licensed real-world
  dataset once one is available, and re-run the full pipeline.
- Add hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`)
  for each candidate model.
- Add SHAP-based explainability for individual predictions.
- Add a lightweight web front-end (e.g. Streamlit) over `predict.py`.
- Track experiments over time with a tool such as MLflow.

## Ethical Considerations

This project predicts *categories of academic performance* from
behavioural and demographic-adjacent data, which raises real ethical
concerns if it were ever applied to real students:

- **Privacy:** any real deployment must use data collected with
  informed consent, appropriate anonymization, and compliance with
  applicable student-data-protection law (e.g. FERPA in the US, or
  local equivalents).
- **Bias and fairness:** features like `socioeconomic_index` or
  `parental_support_level` can encode or amplify existing social
  inequities. A model trained on biased historical data will
  reproduce that bias; fairness auditing (e.g. per-group performance
  breakdowns) should be mandatory before any real use.
- **Responsible interpretation:** a predicted category is a
  probabilistic signal from a handful of features, not a judgment of
  a student's ability, effort, or worth. It should only ever be used
  to *offer support*, never to gatekeep opportunities, track students
  into lower expectations, or replace a human educator's judgment.
- **Limitations of predictive models:** this model's ~78% test
  accuracy means roughly 1 in 5 predictions on held-out data are
  wrong. Any real deployment needs a human in the loop and a clear
  process for a student to contest or ignore the prediction.

## License

MIT License — see `LICENSE`.

## Author

**Cindhe Sai Mukesh Rao**
