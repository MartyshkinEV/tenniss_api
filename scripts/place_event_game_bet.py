try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import gzip
import json
import time
from urllib.request import Request, urlopen

from config import settings
from src.live.fonbet import FonbetBetExecutor, extract_markets_from_fonbet_catalog
from src.live.runtime import LiveMarket, ScoredSelection


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


def _pick_market(payload: dict, target_game: int) -> LiveMarket:
    markets = [
        market
        for market in extract_markets_from_fonbet_catalog(payload)
        if market.market_type == "next_game_winner"
        and int(market.raw.get("target_game_number") or 0) == target_game
    ]
    if not markets:
        available = sorted(
            {
                int(market.raw.get("target_game_number") or 0)
                for market in extract_markets_from_fonbet_catalog(payload)
                if market.market_type == "next_game_winner"
            }
        )
        raise SystemExit(
            "No next_game_winner market found for the requested game"
            + (f"; available games now: {available}" if available else "")
        )
    return markets[0]


def _build_candidate(market: LiveMarket, side: str, stake: float) -> ScoredSelection:
    if side == "player1":
        player_name = market.player1_name
        odds = market.player1_odds
    else:
        player_name = market.player2_name
        odds = market.player2_odds
    implied_probability = 1.0 / odds
    return ScoredSelection(
        market=market,
        side=side,
        player_name=player_name,
        model_probability=implied_probability,
        implied_probability=implied_probability,
        edge=0.0,
        odds=odds,
        stake=stake,
        player_id=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", type=int, required=True)
    parser.add_argument("--target-game", type=int, required=True)
    parser.add_argument("--side", choices=["player1", "player2"], required=True)
    parser.add_argument("--stake", type=float, default=30.0)
    parser.add_argument("--wait-open-seconds", type=float, default=0.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--send-bet", action="store_true")
    args = parser.parse_args()

    executor = FonbetBetExecutor(dry_run=not args.send_bet)
    deadline = time.time() + max(args.wait_open_seconds, 0.0)
    market = None
    candidate = None
    refreshed_selection = {}
    slip_info_response = {}

    while True:
        payload = _load_event_payload(args.event_id)
        try:
            market = _pick_market(payload, args.target_game)
            candidate = _build_candidate(market, args.side, args.stake)
            refreshed_selection, slip_info_response = executor.refresh_candidate(candidate)
            if refreshed_selection.get("value") not in (None, "", 0, 0.0):
                break
        except SystemExit as exc:
            refreshed_selection = {}
            slip_info_response = {"status": "market_not_available", "message": str(exc)}
        if time.time() >= deadline:
            break
        time.sleep(max(args.poll_interval, 0.1))

    if market is None or candidate is None:
        raise SystemExit("Failed to build a live game candidate")

    result = executor.place_prepared_bet(candidate, refreshed_selection, slip_info_response)
    output = {
        "event_id": args.event_id,
        "target_game": args.target_game,
        "side": args.side,
        "stake": args.stake,
        "player1_name": market.player1_name,
        "player2_name": market.player2_name,
        "market_id": market.market_id,
        "market_raw": market.raw,
        "refreshed_selection": refreshed_selection,
        "bet_slip_info_response": slip_info_response,
        "result": result,
    }
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
