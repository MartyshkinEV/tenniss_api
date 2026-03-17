"""Training entrypoint wrappers for extensible model training."""


def train_logreg_baseline():
    from scripts.train_match_model import main

    return main()


def train_lightgbm_baseline():
    from scripts.train_lightgbm_model import main

    return main()


def train_logreg_elo():
    from scripts.train_logreg_elo_model import main

    return main()


def train_lightgbm_elo():
    from scripts.train_lightgbm_elo_model import main

    return main()


def train_game_markov():
    from scripts.train_game_markov_model import main

    return main()


def train_rl_policy():
    from scripts.train_rl_policy import main

    return main()


def train_point_outcome():
    from scripts.train_point_outcome_model import main

    return main()


def train_historical_point():
    from scripts.train_historical_point_model import main

    return main()


def train_historical_game():
    from scripts.train_historical_game_model import main

    return main()


def train_historical_total():
    from scripts.train_historical_total_model import main

    return main()


def train_execution_survival():
    from scripts.train_execution_survival_model import main

    return main()


def train_leg_acceptance():
    from scripts.train_leg_acceptance_model import main

    return main()
