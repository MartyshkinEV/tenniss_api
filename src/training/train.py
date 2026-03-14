"""Training entrypoint wrappers for extensible model training."""


def train_logreg_baseline():
    from scripts.train_match_model import main

    return main()


def train_lightgbm_baseline():
    from scripts.train_lightgbm_model import main

    return main()
