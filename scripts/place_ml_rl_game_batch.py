try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import gzip
import json
from urllib.request import Request, urlopen

import joblib

from config import settings
from src.live.fonbet import FonbetBetExecutor, extract_markets_from_fonbet_catalog
from src.live.game_model import LayeredGamePredictor
from src.live.markov import MarkovGameModel
from src.live.policy import BankrollBanditPolicy
from src.live.runtime import RuntimeConfig, build_candidate_options


def _load_event_payload(event_id: int) -> dict:
    url = (
        f"https://line-lb61-w.bk6bba-resources.com/ma/events/event"
        f"?lang={settings.fonbet_lang}&version=0&eventId={event_id}&scopeMarket={settings.fonbet_scope_market_id}"
    )
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=settings.fonbet_timeout_seconds) as response:
        raw = response.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def _score_event(event_id: int, config: RuntimeConfig, predictor: LayeredGamePredictor, markov_model: MarkovGameModel):
    payload = _load_event_payload(event_id)
    markets = [
        market
        for market in extract_markets_from_fonbet_catalog(payload)
        if market.market_type == "next_game_winner"
    ]
    best = None
    for market in markets:
        prediction = predictor.predict(
            market=market,
            markov_probability=markov_model.predict_next_game({}, market.raw),
        )
        player1_probability = prediction.player1_probability
        player2_probability = 1.0 - player1_probability
        candidates = build_candidate_options(
            ("player1", market.player1_name, player1_probability, market.player1_odds, 0),
            ("player2", market.player2_name, player2_probability, market.player2_odds, 0),
            config=config,
            market=market,
        )
        for candidate in candidates:
            item = {
                "candidate": candidate,
                "market": market,
                "player1_probability": player1_probability,
                "player2_probability": player2_probability,
                "historical_probability": prediction.historical_probability,
                "markov_probability": prediction.markov_probability,
            }
            if best is None or candidate.edge > best["candidate"].edge:
                best = item
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", type=int, action="append", required=True)
    parser.add_argument("--send-bet", action="store_true")
    args = parser.parse_args()

    config = RuntimeConfig.from_settings()
    historical_model_path = settings.models_dir / "historical_game_model.joblib"
    predictor = LayeredGamePredictor(
        game_model=joblib.load(historical_model_path) if historical_model_path.exists() else None,
        game_model_weight=config.game_model_weight,
        markov_weight=config.game_markov_weight,
    )
    markov_model = MarkovGameModel()
    rl_policy = BankrollBanditPolicy(
        outcomes_path=config.rl_outcomes_path,
        bankroll=config.bankroll,
    )
    executor = FonbetBetExecutor(dry_run=not args.send_bet)

    results = []
    for event_id in args.event_id:
        scored = _score_event(event_id, config, predictor, markov_model)
        if scored is None:
            results.append({"event_id": event_id, "status": "no_candidate"})
            continue
        rl_candidate = rl_policy.recommend([scored["candidate"]], "next_game_winner")
        if rl_candidate is None:
            results.append({"event_id": event_id, "status": "rl_no_bet"})
            continue
        rl_candidate = type(rl_candidate)(
            market=rl_candidate.market,
            side=rl_candidate.side,
            player_name=rl_candidate.player_name,
            model_probability=rl_candidate.model_probability,
            implied_probability=rl_candidate.implied_probability,
            edge=rl_candidate.edge,
            odds=rl_candidate.odds,
            stake=rl_candidate.stake,
            player_id=rl_candidate.player_id,
            acceptance_probability=1.0,
            ranking_score=rl_candidate.edge,
        )
        refreshed_selection, slip_info_response = executor.refresh_candidate(rl_candidate)
        result = executor.place_prepared_bet(rl_candidate, refreshed_selection, slip_info_response)
        results.append(
            {
                "event_id": event_id,
                "status": result.get("status"),
                "market_id": rl_candidate.market.market_id,
                "target_game_number": rl_candidate.market.raw.get("target_game_number"),
                "selection_side": rl_candidate.side,
                "selection_name": rl_candidate.player_name,
                "stake": rl_candidate.stake,
                "odds_before_refresh": rl_candidate.odds,
                "model_probability": rl_candidate.model_probability,
                "implied_probability": rl_candidate.implied_probability,
                "edge": rl_candidate.edge,
                "historical_probability": scored["historical_probability"],
                "markov_probability": scored["markov_probability"],
                "player1_probability": scored["player1_probability"],
                "player2_probability": scored["player2_probability"],
                "refreshed_selection": refreshed_selection,
                "bet_slip_info_response": slip_info_response,
                "bet_execution": result,
            }
        )

    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
