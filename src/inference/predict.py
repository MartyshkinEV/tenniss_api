"""Inference entrypoint wrappers for extensible prediction flows."""


def predict_baseline_to_table():
    from scripts.predict_match_model import main

    return main()
