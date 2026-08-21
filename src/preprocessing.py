"""
preprocessing.py

Builds leakage-safe preprocessing pipelines for the student-performance
dataset and provides helpers to split raw data into features/target.

Design notes
------------
* All fitting (imputation statistics, scaling parameters, encoder
  categories) happens exclusively on the training split. The returned
  `ColumnTransformer` is unfitted; callers must call `.fit` only on
  training data and `.transform` on validation/test/prediction data.
* Numerical features are median-imputed then standard-scaled.
* Categorical features are mode-imputed then one-hot encoded.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw dataframe into a feature matrix and target series.

    Identifier and leakage-prone columns (see
    `config.NON_FEATURE_COLUMNS`) are dropped from the feature matrix.
    """
    feature_columns = [
        column for column in df.columns if column not in config.NON_FEATURE_COLUMNS
    ]
    x = df[feature_columns].copy()
    y = df[config.TARGET_COLUMN].copy()
    return x, y


def build_preprocessing_pipeline(
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Construct an unfitted preprocessing `ColumnTransformer`.

    The transformer must be fit only on training data to avoid data
    leakage from validation/test data into imputation statistics or
    scaling parameters.
    """
    if numerical_features is None:
        numerical_features = config.NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = config.CATEGORICAL_FEATURES

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return human-readable feature names after a fitted transform.

    Must be called on a `preprocessor` that has already been fit.
    """
    return list(preprocessor.get_feature_names_out())
