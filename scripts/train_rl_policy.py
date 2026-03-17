try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import json

from config import settings
from src.live.policy import BankrollBanditPolicy


def main():
    policy = BankrollBanditPolicy(
        outcomes_path=settings.live_rl_outcomes_path,
        bankroll=settings.live_bankroll,
        min_samples=1,
    )
    stats = policy._load_stats()
    output_path = settings.models_dir / "rl_bankroll_policy.json"
    output_path.write_text(
        json.dumps(
            {
                "model_type": "bankroll_bandit_policy",
                "bankroll": settings.live_bankroll,
                "stake_levels": list(BankrollBanditPolicy.STAKE_LEVELS),
                "arms": stats,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_path)
    return output_path


if __name__ == "__main__":
    main()
