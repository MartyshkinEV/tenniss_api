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
from src.live.runtime import HistoricalLookup, RuntimeConfig, build_candidate_options


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", type=int, required=True)
    parser.add_argument("--target-game", type=int)
    parser.add_argument("--stake", type=float, default=30.0)
    parser.add_argument("--send-bet", action="store_true")
    args = parser.parse_args()

    payload = _load_event_payload(args.event_id)
    markets = [
        market
        for market in extract_markets_from_fonbet_catalog(payload)
        if market.market_type == "next_game_winner"
    ]
    if args.target_game is not None:
        markets = [market for market in markets if int(market.raw.get("target_game_number") or 0) == args.target_game]
    if not markets:
        raise SystemExit("No next_game_winner markets found for the requested event")

    config = RuntimeConfig.from_settings()
    lookup = HistoricalLookup()
    markov_model = MarkovGameModel()
    historical_model_path = settings.models_dir / "historical_game_model.joblib"
    predictor = LayeredGamePredictor(
        game_model=joblib.load(historical_model_path) if historical_model_path.exists() else None,
        game_model_weight=config.game_model_weight,
        markov_weight=config.game_markov_weight,
    )

    best = None
    for market in markets:
        frame, features, player1_id, player2_id = lookup.build_prediction_frame(market)
        prediction = predictor.predict(
            market=market,
            markov_probability=markov_model.predict_next_game(frame.iloc[0].to_dict(), market.raw),
        )
        player1_probability = prediction.player1_probability
        player2_probability = 1.0 - player1_probability
        candidates = build_candidate_options(
            ("player1", market.player1_name, player1_probability, market.player1_odds, player1_id),
            ("player2", market.player2_name, player2_probability, market.player2_odds, player2_id),
            config=config,
            market=market,
        )
        for candidate in candidates:
            candidate = type(candidate)(
                market=candidate.market,
                side=candidate.side,
                player_name=candidate.player_name,
                model_probability=candidate.model_probability,
                implied_probability=candidate.implied_probability,
                edge=candidate.edge,
                odds=candidate.odds,
                stake=args.stake,
                player_id=candidate.player_id,
                acceptance_probability=1.0,
                ranking_score=candidate.edge,
            )
            item = {
                "market": market,
                "candidate": candidate,
                "player1_probability": player1_probability,
                "player2_probability": player2_probability,
                "historical_probability": prediction.historical_probability,
                "markov_probability": prediction.markov_probability,
            }
            if best is None or item["candidate"].edge > best["candidate"].edge:
                best = item

    if best is None:
        raise SystemExit("Fast ML analysis did not produce a valid next_game_winner candidate")

    candidate = best["candidate"]
    executor = FonbetBetExecutor(dry_run=not args.send_bet)
    refreshed_selection, slip_info_response = executor.refresh_candidate(candidate)
    result = executor.place_prepared_bet(candidate, refreshed_selection, slip_info_response)
    output = {
        "event_id": args.event_id,
        "market_id": candidate.market.market_id,
        "target_game_number": candidate.market.raw.get("target_game_number"),
        "selection_side": candidate.side,
        "selection_name": candidate.player_name,
        "odds_before_refresh": candidate.odds,
        "model_probability": candidate.model_probability,
        "implied_probability": candidate.implied_probability,
        "edge": candidate.edge,
        "historical_probability": best["historical_probability"],
        "markov_probability": best["markov_probability"],
        "player1_probability": best["player1_probability"],
        "player2_probability": best["player2_probability"],
        "refreshed_selection": refreshed_selection,
        "bet_slip_info_response": slip_info_response,
        "bet_execution": result,
    }
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
