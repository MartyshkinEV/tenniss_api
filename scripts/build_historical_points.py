try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

from src.data import load_historical_points


def main() -> None:
    points = load_historical_points(refresh=True)
    print(
        {
            "rows": len(points),
            "matches": int(points["pbp_id"].nunique()) if not points.empty else 0,
            "date_min": None if points.empty else str(points["date"].min().date()),
            "date_max": None if points.empty else str(points["date"].max().date()),
        }
    )


if __name__ == "__main__":
    main()
