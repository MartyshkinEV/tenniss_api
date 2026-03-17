try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import json
from pathlib import Path

from config import settings


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--progress-path",
        default=str(settings.artifacts_dir / "imports" / "pointbypoint_progress.json"),
    )
    args = parser.parse_args()

    path = Path(args.progress_path).resolve()
    if not path.exists():
        print({"status": "missing", "progress_path": str(path)})
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    print(
        {
            "status": payload.get("status"),
            "percent_complete": payload.get("percent_complete"),
            "processed_files": payload.get("processed_files"),
            "total_files": payload.get("total_files"),
            "processed_matches": payload.get("processed_matches"),
            "total_matches": payload.get("total_matches"),
            "total_points_inserted": payload.get("total_points_inserted"),
            "current_file": payload.get("current_file"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "eta": format_eta(payload.get("eta_seconds")),
            "message": payload.get("message"),
        }
    )


if __name__ == "__main__":
    main()
