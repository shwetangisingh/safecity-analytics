import pickle
import sys
from pathlib import Path
from typing import Any

# ── MCP import ────────────────────────────────────────────────────────────────
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    import mcp.types as types
except ImportError:
    print("ERROR: mcp package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# ── Paths / constants ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "random_forest_model.pkl"
FALLBACK_MODEL_PATH = ROOT / "models" / "decision_tree_model.pkl"
CLEANED_DATA_PATH = ROOT / "data" / "processed" / "crime_data_cleaned.csv"
RANDOM_SEED = 42
AREA_MIN, AREA_MAX = 1, 21
HOUR_MIN, HOUR_MAX = 0, 23
MONTH_MIN, MONTH_MAX = 1, 12


def _validate_bundle(bundle: dict[str, Any], source: Path) -> None:
    required = {"model", "encoders", "feature_names"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"{source} missing required keys: {sorted(missing)}")
    if not hasattr(bundle["model"], "predict"):
        raise ValueError(f"{source} model object has no predict()")
    if not hasattr(bundle["model"], "predict_proba"):
        raise ValueError(f"{source} model object has no predict_proba()")


def _load_bundle_from_path(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    _validate_bundle(bundle, path)
    return bundle


def _train_fallback_random_forest() -> dict[str, Any]:
    """
    Build a compact, inference-ready RandomForest bundle if the serialized model
    is missing or corrupted (common with partial LFS downloads).
    """
    print("Building fallback Random Forest model for MCP server...", file=sys.stderr)

    # Local imports keep startup dependencies minimal unless fallback is needed.
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    model_src = ROOT / "src" / "models"
    if str(model_src) not in sys.path:
        sys.path.append(str(model_src))
    from preprocess import load_data, get_classification_features  # noqa: WPS433

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {CLEANED_DATA_PATH}. "
            "Run src/data_cleaning.py first."
        )

    df = load_data(str(CLEANED_DATA_PATH))
    X, y_category, _, feature_names, encoders = get_classification_features(df)

    # Align with training scripts by dropping ultra-rare classes.
    class_counts = pd.Series(y_category).value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    mask = np.isin(y_category, valid_classes)
    X = X[mask]
    y_category = y_category[mask]

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X, y_category)

    bundle = {
        "model": rf,
        "encoders": encoders,
        "feature_names": feature_names,
        "best_params": {
            "source": "server_fallback_training",
            "random_state": RANDOM_SEED,
            "n_estimators": 150,
            "max_depth": 20,
            "min_samples_split": 10,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved rebuilt model bundle -> {MODEL_PATH}", file=sys.stderr)
    return bundle


def _load_or_rebuild_bundle() -> tuple[dict[str, Any], str]:
    if MODEL_PATH.exists():
        try:
            bundle = _load_bundle_from_path(MODEL_PATH)
            print(f"Loaded model bundle from {MODEL_PATH}", file=sys.stderr)
            return bundle, str(MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: Could not load {MODEL_PATH} ({exc}). Attempting rebuild...",
                file=sys.stderr,
            )

    try:
        return _train_fallback_random_forest(), "fallback_retrain"
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: Fallback training failed: {exc}", file=sys.stderr)

    if FALLBACK_MODEL_PATH.exists():
        try:
            bundle = _load_bundle_from_path(FALLBACK_MODEL_PATH)
            print(
                f"Using decision-tree fallback bundle from {FALLBACK_MODEL_PATH}",
                file=sys.stderr,
            )
            return bundle, str(FALLBACK_MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: Could not load {FALLBACK_MODEL_PATH} ({exc})",
                file=sys.stderr,
            )

    print(
        "ERROR: No usable model bundle available. "
        "Run src/models/train_random_forest.py to generate one.",
        file=sys.stderr,
    )
    sys.exit(1)


bundle, model_source = _load_or_rebuild_bundle()
model = bundle["model"]
encoders = bundle["encoders"]
feature_names = bundle["feature_names"]

# Pre-extract encoder maps for validation
premise_classes = list(encoders["Premise Category"].classes_)
timebucket_classes = list(encoders["TimeBucket"].classes_)
severity_classes = list(encoders["Severity"].classes_)
crime_classes = list(encoders["Crime Category"].classes_)

# ── MCP Server ────────────────────────────────────────────────────────────────
server = Server("safecity-crime-predictor")


def _text(message: str):
    return [types.TextContent(type="text", text=message)]


def _validate_choice(value: str, valid_values: list[str], label: str) -> str | None:
    if value not in valid_values:
        return f"Invalid {label}: '{value}'. Must be one of {valid_values}"
    return None


def _format_top_predictions(probabilities, top_indices):
    top3 = []
    for idx in top_indices:
        class_id = int(model.classes_[idx])
        class_name = encoders["Crime Category"].inverse_transform([class_id])[0]
        top3.append(f"{class_name}: {probabilities[idx]*100:.1f}%")
    return top3


@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="predict_crime_category",
            description=(
                "Predict the most likely crime category for a given incident context "
                "using a trained Random Forest model (SafeCity Phase 2)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "area": {
                        "type": "integer",
                        "description": "LAPD area code (1–21)",
                        "minimum": AREA_MIN, "maximum": AREA_MAX
                    },
                    "hour": {
                        "type": "integer",
                        "description": "Hour of day (0–23)",
                        "minimum": HOUR_MIN, "maximum": HOUR_MAX
                    },
                    "month": {
                        "type": "integer",
                        "description": "Month (1–12)",
                        "minimum": MONTH_MIN, "maximum": MONTH_MAX
                    },
                    "is_weekend": {
                        "type": "boolean",
                        "description": "True if the incident occurred on a weekend"
                    },
                    "has_weapon": {
                        "type": "boolean",
                        "description": "True if a weapon was involved"
                    },
                    "premise_category": {
                        "type": "string",
                        "description": f"One of: {premise_classes}",
                        "enum": premise_classes
                    },
                    "time_bucket": {
                        "type": "string",
                        "description": f"One of: {timebucket_classes}",
                        "enum": timebucket_classes
                    },
                    "severity": {
                        "type": "string",
                        "description": f"One of: {severity_classes}",
                        "enum": severity_classes
                    },
                    "part_1_2": {
                        "type": "integer",
                        "description": "Part 1 or Part 2 crime (1 or 2)",
                        "enum": [1, 2]
                    },
                    "reporting_delay_days": {
                        "type": "integer",
                        "description": "Number of days between crime occurrence and report",
                        "minimum": 0
                    },
                },
                "required": [
                    "area", "hour", "month", "is_weekend", "has_weapon",
                    "premise_category", "time_bucket", "severity",
                    "part_1_2", "reporting_delay_days"
                ]
            }
        ),
        types.Tool(
            name="list_crime_categories",
            description="List all crime categories the model can predict.",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="server_health",
            description="Return model/source metadata for MCP diagnostics.",
            inputSchema={"type": "object", "properties": {}}
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "server_health":
        return _text(
            "SafeCity MCP server is healthy.\n"
            f"Model source: {model_source}\n"
            f"Classes available: {len(getattr(model, 'classes_', []))}\n"
            f"Feature count: {len(feature_names)}"
        )

    if name == "list_crime_categories":
        categories = sorted(crime_classes)
        return _text(
            f"Predictable crime categories ({len(categories)}):\n"
            + "\n".join(f"  - {c}" for c in categories)
        )

    if name == "predict_crime_category":
        # ── input validation ──────────────────────────────────────────────────
        try:
            area = int(arguments["area"])
            hour = int(arguments["hour"])
            month = int(arguments["month"])
            is_weekend = int(bool(arguments["is_weekend"]))
            has_weapon = int(bool(arguments["has_weapon"]))
            premise = arguments["premise_category"]
            timebucket = arguments["time_bucket"]
            severity = arguments["severity"]
            part = int(arguments["part_1_2"])
            delay = int(arguments["reporting_delay_days"])
        except (KeyError, ValueError) as e:
            return _text(f"Input error: {e}")

        if not (AREA_MIN <= area <= AREA_MAX):
            return _text(f"Input error: area must be {AREA_MIN}..{AREA_MAX}")
        if not (HOUR_MIN <= hour <= HOUR_MAX):
            return _text(f"Input error: hour must be {HOUR_MIN}..{HOUR_MAX}")
        if not (MONTH_MIN <= month <= MONTH_MAX):
            return _text(f"Input error: month must be {MONTH_MIN}..{MONTH_MAX}")
        if part not in (1, 2):
            return _text("Input error: part_1_2 must be 1 or 2")
        if delay < 0:
            return _text("Input error: reporting_delay_days must be >= 0")

        # Validate categorical values
        for val, valid_list, label in [
            (premise, premise_classes, "premise_category"),
            (timebucket, timebucket_classes, "time_bucket"),
            (severity, severity_classes, "severity"),
        ]:
            error = _validate_choice(val, valid_list, label)
            if error:
                return _text(error)

        # ── encode categoricals ───────────────────────────────────────────────
        premise_enc  = encoders["Premise Category"].transform([premise])[0]
        time_enc     = encoders["TimeBucket"].transform([timebucket])[0]
        severity_enc = encoders["Severity"].transform([severity])[0]

        features = [[
            area, hour, month, is_weekend, has_weapon,
            premise_enc, time_enc, severity_enc, part, delay
        ]]

        # ── predict ───────────────────────────────────────────────────────────
        pred_idx = model.predict(features)[0]
        pred_label = encoders["Crime Category"].inverse_transform([pred_idx])[0]
        probas = model.predict_proba(features)[0]
        top3_idx = probas.argsort()[-3:][::-1]
        top3 = _format_top_predictions(probas, top3_idx)

        result = (
            f"**Predicted Crime Category:** {pred_label}\n\n"
            f"**Top 3 Predictions:**\n" + "\n".join(f"  {i+1}. {t}" for i, t in enumerate(top3))
        )
        return _text(result)

    return _text(f"Unknown tool: {name}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream,
                             server.create_initialization_options())

    asyncio.run(main())
