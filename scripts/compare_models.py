try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from config import (
    MODEL_SPECS,
    get_model_spec,
    resolve_default_model_artifact_path,
    resolve_default_model_name,
    resolve_model_artifact_path,
    settings,
)
from src.data import BASELINE_FEATURES, ELO_FEATURES, align_frame_to_model
from src.data.pipeline import load_match_features, load_match_features_elo, split_dataset


def _dataset_for_model(model_name: str):
    spec = get_model_spec(model_name)
    if spec["feature_set"] == "elo":
        return load_match_features_elo(), ELO_FEATURES
    return load_match_features(), BASELINE_FEATURES


def collect_metrics() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name in MODEL_SPECS:
        model_path = resolve_model_artifact_path(model_name)
        model = joblib.load(model_path)
        df, features = _dataset_for_model(model_name)
        splits = split_dataset(df)
        valid_X = align_frame_to_model(model, splits["valid"], features)
        test_X = align_frame_to_model(model, splits["test"], features)
        valid_y = splits["valid"]["label"]
        test_y = splits["test"]["label"]
        valid_proba = model.predict_proba(valid_X)[:, 1]
        test_proba = model.predict_proba(test_X)[:, 1]
        rows.append(
            {
                "model_name": model_name.removesuffix(".joblib"),
                "artifact_path": str(model_path),
                "feature_set": get_model_spec(model_name)["feature_set"],
                "valid_auc": float(roc_auc_score(valid_y, valid_proba)),
                "test_auc": float(roc_auc_score(test_y, test_proba)),
                "test_log_loss": float(log_loss(test_y, test_proba)),
                "reference_valid_auc": float(get_model_spec(model_name)["valid_auc"]),
                "reference_test_auc": float(get_model_spec(model_name)["test_auc"]),
                "reference_test_log_loss": float(get_model_spec(model_name)["test_log_loss"]),
            }
        )
    rows.sort(key=lambda row: (-float(row["test_auc"]), float(row["test_log_loss"])))
    return rows


def render_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "model_name",
        "feature_set",
        "valid_auc",
        "test_auc",
        "test_log_loss",
        "delta_test_auc",
        "delta_test_log_loss",
    ]
    table_rows = []
    for row in rows:
        rendered = dict(row)
        rendered["delta_test_auc"] = float(row["test_auc"]) - float(row["reference_test_auc"])
        rendered["delta_test_log_loss"] = float(row["test_log_loss"]) - float(row["reference_test_log_loss"])
        table_rows.append(rendered)
    widths = {
        header: max(
            len(header),
            *(
                len(f"{row[header]:.6f}") if isinstance(row[header], float) else len(str(row[header]))
                for row in table_rows
            ),
        )
        for header in headers
    }
    lines = [
        "  ".join(header.ljust(widths[header]) for header in headers),
        "  ".join("-" * widths[header] for header in headers),
    ]
    for row in table_rows:
        rendered = []
        for header in headers:
            value = row[header]
            cell = f"{value:.6f}" if isinstance(value, float) else str(value)
            rendered.append(cell.ljust(widths[header]))
        lines.append("  ".join(rendered))
    return "\n".join(lines)


def write_report(rows: list[dict[str, object]], report_path: Path, json_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    recommended = rows[0]["model_name"] if rows else resolve_default_model_name().removesuffix(".joblib")
    lines = [
        "# Model Report",
        "",
        "## Summary",
        f"- Recommended default model: `{recommended}`",
        f"- Production artifact: `{resolve_default_model_name()}`",
        f"- Production path used by prediction commands: `{resolve_default_model_artifact_path()}`",
        "",
        "## Metrics",
        "",
        "| model | feature_set | valid_auc | test_auc | test_log_loss | delta_test_auc | delta_test_log_loss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {row['feature_set']} | {row['valid_auc']:.6f} | {row['test_auc']:.6f} | {row['test_log_loss']:.6f} | "
            f"{row['test_auc'] - row['reference_test_auc']:+.6f} | {row['test_log_loss'] - row['reference_test_log_loss']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Production Commands",
            f"- Point prediction: `./venv/bin/python scripts/predict_match.py 106168 210000`",
            f"- Batch prediction: `./venv/bin/python scripts/predict_match_model.py`",
            f"- Retrain recommended default model: `./venv/bin/python scripts/train_lightgbm_elo_model.py`",
            f"- Compare trained models: `./venv/bin/python scripts/compare_models.py`",
            "",
            "## Notes",
            "- Batch prediction now defaults to the same production model as point prediction.",
            "- Metrics above are recomputed from the local cached dataset/model artifacts in this repository.",
            "- If a frozen legacy artifact exists for the production default, prediction commands prefer it over newly retrained candidates.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_payload = {
        "recommended_default_model": recommended,
        "production_default_artifact": resolve_default_model_name(),
        "models": rows,
    }
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    rows = collect_metrics()
    print(render_table(rows))

    if args.write_report:
        report_dir = settings.artifacts_dir / "reports"
        write_report(
            rows,
            report_dir / "model_report.md",
            report_dir / "model_metrics.json",
        )
        print()
        print(f"Wrote report to {report_dir / 'model_report.md'}")
        print(f"Wrote metrics to {report_dir / 'model_metrics.json'}")


if __name__ == "__main__":
    main()
