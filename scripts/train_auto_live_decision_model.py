try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from config import settings
from src.training.live_point_models import train_binary_classifier


DEFAULT_INPUT = settings.artifacts_dir / "live_betting" / "auto_live_training_queue.jsonl"
DEFAULT_MODEL = settings.models_dir / "auto_live_decision_model.joblib"
DEFAULT_META = settings.models_dir / "auto_live_decision_model.json"


def _status_to_label(status: str) -> int | None:
    if status == "placed":
        return 1
    if status in {"no_candidate", "rl_no_bet", "exposure_blocked", "event_failed"}:
        return 0
    return None


def _load_rows(path: Path, max_rows: int | None) -> list[dict]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_rows is not None and max_rows > 0:
        lines = lines[-max_rows:]
    rows: list[dict] = []
    for line in lines:
        payload = json.loads(line)
        label = _status_to_label(str(payload.get("status") or ""))
        if label is None:
            continue
        rows.append(
            {
                "event_id": int(payload.get("event_id") or 0),
                "market_id": str(payload.get("market_id") or ""),
                "target_game_number": payload.get("target_game_number"),
                "selection_side": str(payload.get("selection_side") or ""),
                "stake": payload.get("stake"),
                "odds_before_refresh": payload.get("odds_before_refresh"),
                "model_probability": payload.get("model_probability"),
                "implied_probability": payload.get("implied_probability"),
                "edge": payload.get("edge"),
                "historical_probability": payload.get("historical_probability"),
                "markov_probability": payload.get("markov_probability"),
                "player1_probability": payload.get("player1_probability"),
                "player2_probability": payload.get("player2_probability"),
                "score": ((payload.get("refreshed_selection") or {}).get("score")),
                "param": ((payload.get("refreshed_selection") or {}).get("param")),
                "factor_id": ((payload.get("refreshed_selection") or {}).get("factor_id")),
                "label": label,
            }
        )
    return rows


def main() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_MODEL))
    parser.add_argument("--metadata-output", default=str(DEFAULT_META))
    parser.add_argument("--max-rows", type=int, default=5000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)

    rows = _load_rows(input_path, args.max_rows)
    frame = pd.DataFrame(rows)
    result = train_binary_classifier(frame, target_name="auto_live_decision")
    metadata = dict(result.metadata)
    metadata.update(
        {
            "input_path": str(input_path),
            "rows_loaded": int(len(frame)),
            "max_rows": args.max_rows,
            "labels": {
                "placed": 1,
                "no_candidate|rl_no_bet|exposure_blocked|event_failed": 0,
            },
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, output_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    print(output_path)
    print(metadata_path)
    return str(output_path)


if __name__ == "__main__":
    main()
