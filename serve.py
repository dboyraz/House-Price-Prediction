import os
from pathlib import Path
from typing import List

import pandas as pd
from flask import Flask, jsonify, request

from predict import (
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    HousePricePredictor,
    load_model_artifact,
)

MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))

app = Flask(__name__)

try:
    _MODEL_BUNDLE = load_model_artifact(MODEL_PATH)
    _PREDICTOR = HousePricePredictor(
        _MODEL_BUNDLE["model"],
        _MODEL_BUNDLE.get("feature_names", FEATURE_COLUMNS),
    )
except FileNotFoundError:
    _MODEL_BUNDLE = None
    _PREDICTOR = None


def _normalize_payload(payload) -> List[dict]:
    if payload is None:
        raise ValueError("Request body must be valid JSON.")

    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        records = payload

    if not isinstance(records, list):
        raise ValueError("JSON payload must be an object or list of objects.")
    if not records:
        raise ValueError("No records provided.")
    return records


@app.route("/health", methods=["GET"])
def health():
    status = "ready" if _PREDICTOR is not None else "model_missing"
    return jsonify(
        {
            "status": status,
            "model_path": str(MODEL_PATH),
            "metrics": (_MODEL_BUNDLE or {}).get("metrics"),
        }
    )


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if _PREDICTOR is None:
        return (
            jsonify({"error": f"Model artifact not found at {MODEL_PATH}."}),
            503,
        )

    try:
        records = _normalize_payload(request.get_json())
        df = pd.DataFrame.from_records(records)
        predictions = _PREDICTOR.predict(df)["predictions"]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive programming
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500

    return jsonify({"predictions": predictions, "count": len(predictions)})


def main() -> None:
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


if __name__ == "__main__":
    main()
