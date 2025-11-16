import argparse
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from predict import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_matrix,
    prepare_dataframe,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "AmesHousing.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {csv_path}")
    return pd.read_csv(csv_path)


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    max_features=0.3,
                    min_samples_leaf=1,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_predictions(
    y_true_log: np.ndarray, y_pred_log: np.ndarray
) -> Dict[str, float]:
    rmse_log = root_mean_squared_error(y_true_log, y_pred_log)
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log)

    y_true_price = np.expm1(y_true_log)
    y_pred_price = np.expm1(y_pred_log)

    rmse = root_mean_squared_error(y_true_price, y_pred_price)
    mae = mean_absolute_error(y_true_price, y_pred_price)
    r2 = r2_score(y_true_price, y_pred_price)

    return {
        "rmse_log": float(rmse_log),
        "mae_log": float(mae_log),
        "r2_log": float(r2_log),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def train(
    data_path: Path,
    artifact_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, float]]:
    raw_df = load_dataset(data_path)
    processed_df = prepare_dataframe(raw_df)

    if TARGET_COLUMN not in processed_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not created during preprocessing."
        )

    y = processed_df[TARGET_COLUMN].to_numpy()
    X = build_feature_matrix(processed_df, FEATURE_COLUMNS)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = build_model(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred_log)

    artifact = {
        "model": model,
        "feature_names": list(X.columns),
        "target": TARGET_COLUMN,
        "metrics": metrics,
        "training_params": {
            "test_size": test_size,
            "random_state": random_state,
            "n_train_rows": len(X_train),
            "n_test_rows": len(X_test),
        },
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    return model, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Ames Housing price model and save the artifact."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to AmesHousing.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to store the trained model artifact (default: %(default)s)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out size used for evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and model (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, metrics = train(
        data_path=args.data,
        artifact_path=args.artifact,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("Training complete. Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
