import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"

TARGET_COLUMN = "saleprice_log"
FEATURE_COLUMNS: List[str] = [
    # Engineered / transformed features
    "total_sf_log",
    "1st_flr_sf_log1p",
    "total_bsmt_sf_log1p",
    "exter_qual_num",
    "kitchen_qual_num",
    "bsmt_qual_num",
    "fireplace_qu_num",
    "heating_qc_num",
    "garage_qual_num",
    "garage_cond_num",
    "overall_score",
    "house_age",
    "since_remodel",
    "garage_age_log1p",
    "total_bath",
    "bath_per_bedroom",
    "lot_area_log1p",
    "mas_vnr_area_log1p",
    "bsmt_finished_sf",
    "total_outdoor_sf",
    "open_porch_sf_log1p",
    "wood_deck_sf_log1p",
    "has_fireplace",
    "has_deck",
    "has_porch",
    # Raw numerics worth keeping
    "overall_qual",
    "garage_cars",
    "garage_area",
    "total_bsmt_sf",
    "totrms_abvgrd",
]


def _has_columns(df: pd.DataFrame, *cols: str) -> bool:
    """Utility to check column availability before deriving new features."""
    return set(cols).issubset(df.columns)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Match the notebook's column normalization."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")
    )
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values exactly as in the exploratory notebook."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna("None")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the engineered features from the notebook."""
    df = df.copy()

    df = df.drop(
        columns=["utilities", "street", "kitchen_abvgr", "roof_matl", "order", "pid"],
        errors="ignore",
    )

    df = df.rename(columns={"year_remod/add": "year_remod_add"})

    if _has_columns(df, "yr_sold", "year_built"):
        df["house_age"] = df["yr_sold"] - df["year_built"]
    if _has_columns(df, "yr_sold", "year_remod_add"):
        df["since_remodel"] = df["yr_sold"] - df["year_remod_add"]
        df["is_remodeled"] = (df["year_built"] != df["year_remod_add"]).astype(int)

    if "mo_sold" in df.columns:
        df["sale_season"] = pd.cut(
            df["mo_sold"],
            bins=[0, 3, 6, 9, 12],
            labels=["winter", "spring", "summer", "fall"],
            include_lowest=True,
        )

    if _has_columns(df, "total_bsmt_sf", "gr_liv_area"):
        df["total_sf"] = df["total_bsmt_sf"] + df["gr_liv_area"]
    if _has_columns(df, "bsmtfin_sf_1", "bsmtfin_sf_2"):
        df["bsmt_finished_sf"] = df["bsmtfin_sf_1"] + df["bsmtfin_sf_2"]
    porch_cols = ["open_porch_sf", "enclosed_porch", "3ssn_porch", "screen_porch"]
    if _has_columns(df, *porch_cols):
        df["total_porch_sf"] = sum(df[col] for col in porch_cols)
    if _has_columns(df, "wood_deck_sf", "total_porch_sf"):
        df["total_outdoor_sf"] = df["wood_deck_sf"] + df["total_porch_sf"]

    if "lot_area" in df.columns:
        df["lot_area_log"] = np.log1p(df["lot_area"])
    if "gr_liv_area" in df.columns:
        df["gr_liv_area_log"] = np.log1p(df["gr_liv_area"])
    if "total_sf" in df.columns:
        df["total_sf_log"] = np.log1p(df["total_sf"].clip(lower=0))

    bath_cols = ["bsmt_full_bath", "full_bath", "bsmt_half_bath", "half_bath"]
    if _has_columns(df, *bath_cols):
        df["total_bath"] = (
            df["bsmt_full_bath"]
            + df["full_bath"]
            + 0.5 * (df["bsmt_half_bath"] + df["half_bath"])
        )
    if _has_columns(df, "bsmt_full_bath", "full_bath"):
        df["total_full_bath"] = df["bsmt_full_bath"] + df["full_bath"]
    if _has_columns(df, "bedroom_abvgr", "totrms_abvgrd"):
        df["bedroom_per_room"] = df["bedroom_abvgr"] / df["totrms_abvgrd"].replace(
            0, np.nan
        )
    if _has_columns(df, "total_bath", "bedroom_abvgr"):
        df["bath_per_bedroom"] = df["total_bath"] / df["bedroom_abvgr"].replace(
            0, np.nan
        )
    if _has_columns(df, "lot_area", "total_sf"):
        df["lot_area_per_sf"] = df["lot_area"] / df["total_sf"].replace(0, np.nan)
    if _has_columns(df, "gr_liv_area", "total_sf"):
        df["liv_area_ratio"] = df["gr_liv_area"] / df["total_sf"].replace(0, np.nan)

    if "garage_type" in df.columns or "garage_area" in df.columns:
        if "garage_type" in df.columns:
            garage_type_clean = df["garage_type"].replace("None", np.nan)
        else:
            garage_type_clean = pd.Series(np.nan, index=df.index)
        garage_area = (
            df["garage_area"] if "garage_area" in df.columns else pd.Series(0, index=df.index)
        )
        df["has_garage"] = ((~garage_type_clean.isna()) | (garage_area > 0)).astype(int)
    if _has_columns(df, "yr_sold", "garage_yr_blt"):
        df["garage_age"] = df["yr_sold"] - df["garage_yr_blt"]
        df.loc[df["garage_yr_blt"].isna() | (df["garage_yr_blt"] <= 0), "garage_age"] = np.nan

    if "total_bsmt_sf" in df.columns:
        df["has_bsmt"] = (df["total_bsmt_sf"] > 0).astype(int)
    if "fireplaces" in df.columns:
        df["has_fireplace"] = (df["fireplaces"] > 0).astype(int)
    if "pool_area" in df.columns:
        df["has_pool"] = (df["pool_area"] > 0).astype(int)
    if "2nd_flr_sf" in df.columns:
        df["has_2ndfloor"] = (df["2nd_flr_sf"] > 0).astype(int)
    if "wood_deck_sf" in df.columns:
        df["has_deck"] = (df["wood_deck_sf"] > 0).astype(int)
    if "total_porch_sf" in df.columns:
        df["has_porch"] = (df["total_porch_sf"] > 0).astype(int)

    if _has_columns(df, "condition_1", "condition_2"):
        df["same_condition"] = (df["condition_1"] == df["condition_2"]).astype(int)

    if _has_columns(df, "overall_qual", "overall_cond"):
        df["overall_score"] = df["overall_qual"] * df["overall_cond"]
    if "overall_qual" in df.columns:
        df["overall_qual_sq"] = df["overall_qual"] ** 2
    if "overall_cond" in df.columns:
        df["overall_cond_sq"] = df["overall_cond"] ** 2

    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    ord_cols = [
        "exter_qual",
        "exter_cond",
        "bsmt_qual",
        "bsmt_cond",
        "heating_qc",
        "kitchen_qual",
        "fireplace_qu",
        "garage_qual",
        "garage_cond",
        "pool_qc",
    ]
    for col in ord_cols:
        if col in df.columns:
            df[col + "_num"] = df[col].map(qual_map).fillna(0).astype(int)

    rare_cat_cols = ["neighborhood", "exterior_1st", "exterior_2nd", "misc_feature"]
    for col in rare_cat_cols:
        if col in df.columns:
            freqs = df[col].value_counts(normalize=True)
            rare_labels = freqs[freqs < 0.01].index
            df[col + "_grp"] = df[col].replace(rare_labels, "Rare")

    if "saleprice" in df.columns:
        df["saleprice_log"] = np.log1p(df["saleprice"])

    return df


def add_skew_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log_features = [
        "misc_val",
        "pool_area",
        "lot_area",
        "low_qual_fin_sf",
        "3ssn_porch",
        "bsmtfin_sf_2",
        "enclosed_porch",
        "screen_porch",
        "bsmt_half_bath",
        "mas_vnr_area",
        "open_porch_sf",
        "wood_deck_sf",
        "1st_flr_sf",
        "bsmtfin_sf_1",
        "gr_liv_area",
        "total_bsmt_sf",
    ]

    for col in log_features:
        if col in df.columns:
            df[col + "_log1p"] = np.log1p(df[col])

    if {"yr_sold", "garage_yr_blt"}.issubset(df.columns):
        df["garage_age"] = df["yr_sold"] - df["garage_yr_blt"]
        missing_mask = df["garage_yr_blt"].isna() | (df["garage_yr_blt"] <= 0)
        df.loc[missing_mask, "garage_age"] = np.nan
        df["garage_age_log1p"] = np.log1p(df["garage_age"].clip(lower=0))

    if "saleprice" in df.columns and "saleprice_log" not in df.columns:
        df["saleprice_log"] = np.log1p(df["saleprice"])

    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_column_names(df)
    df = fill_missing_values(df)
    df = add_engineered_features(df)
    df = add_skew_transforms(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_feature_matrix(
    df: pd.DataFrame, feature_list: Sequence[str] | None = None
) -> pd.DataFrame:
    features = feature_list or FEATURE_COLUMNS
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(
            "Some required features are missing after preprocessing: "
            + ", ".join(missing)
        )
    return df[features]


def load_model_artifact(model_path: Path):
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle
    return {"model": bundle, "feature_names": FEATURE_COLUMNS, "target": TARGET_COLUMN}


class HousePricePredictor:
    """Thin wrapper that handles preprocessing + sklearn inference."""

    def __init__(self, model, feature_names: Sequence[str] | None = None):
        self.model = model
        self.feature_names = list(feature_names or FEATURE_COLUMNS)

    def predict(self, records: Iterable[dict] | pd.DataFrame) -> dict:
        if isinstance(records, pd.DataFrame):
            df = records.copy()
        else:
            df = pd.DataFrame.from_records(records)
        processed = prepare_dataframe(df)
        X = build_feature_matrix(processed, self.feature_names)
        preds_log = self.model.predict(X)
        preds_price = np.expm1(preds_log)
        return {
            "predictions": [
                {"saleprice": float(p), "saleprice_log": float(lp)}
                for p, lp in zip(preds_price, preds_log)
            ]
        }


def _load_records(input_path: Path) -> List[dict]:
    payload = json.loads(input_path.read_text())
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError("Input JSON must be an object or a list of objects.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the trained house price model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved model artifact (default: %(default)s)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a JSON file containing one record or a list of records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write predictions as JSON. Prints to stdout if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_records(args.input)
    model_bundle = load_model_artifact(args.model)
    predictor = HousePricePredictor(
        model_bundle["model"], model_bundle.get("feature_names", FEATURE_COLUMNS)
    )
    result = predictor.predict(records)
    output_text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
