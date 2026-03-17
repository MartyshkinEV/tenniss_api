try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import gzip
import json
import socket
import time
from urllib.request import Request, urlopen

from config import settings
from src.live.fonbet import FonbetBetExecutor, extract_markets_from_fonbet_catalog
from src.live.runtime import LiveBettingRuntime


class _EmptyFeedClient:
    def fetch_live_markets(self):
        return []


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
    parser.add_argument("--target-game", type=int, help="Prefer a specific target game number.")
    parser.add_argument("--side", choices=["auto", "player1", "player2"], default="auto")
    parser.add_argument("--stake", type=float, default=30.0)
    parser.add_argument("--skip-refresh", action="store_true", help="Print ML-selected pre-refresh payload only.")
    parser.add_argument("--refresh-timeout", type=float, default=8.0, help="Timeout in seconds for betSlipInfo refresh.")
    parser.add_argument("--wait-open-seconds", type=float, default=0.0, help="Keep polling the event until ML finds a valid live candidate.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval while waiting for an ML-valid live candidate.")
    parser.add_argument("--send-bet", action="store_true", help="After successful refresh, send betRequestId -> bet -> betResult.")
    args = parser.parse_args()
    if args.send_bet and args.skip_refresh:
        raise SystemExit("--send-bet cannot be used together with --skip-refresh")

    runtime = LiveBettingRuntime(
        market_feed_client=_EmptyFeedClient(),
        bet_executor=FonbetBetExecutor(dry_run=True),
    )
    best = None
    payload = None
    available_targets = []
    deadline = time.time() + max(args.wait_open_seconds, 0.0)
    while True:
        payload = _load_event_payload(args.event_id)
        all_game_markets = [
            market for market in extract_markets_from_fonbet_catalog(payload)
            if market.market_type == "next_game_winner"
        ]
        markets = list(all_game_markets)
        if args.target_game is not None:
            markets = [
                market for market in markets
                if market.raw.get("target_game_number") == args.target_game
            ]
        available_targets = sorted(
            {
                market.raw.get("target_game_number")
                for market in all_game_markets
                if market.raw.get("target_game_number") is not None
            }
        )
        best = None
        for market in sorted(markets, key=lambda item: int(item.raw.get("target_game_number") or 999999)):
            frame, features, player1_probability, player2_probability, candidate = runtime._score_market_details(market)
            if candidate is None:
                continue
            if args.side != "auto" and candidate.side != args.side:
                continue
            ranked = {
                "market": market,
                "features": features,
                "player1_probability": player1_probability,
                "player2_probability": player2_probability,
                "candidate": candidate,
                "ranking_score": float(getattr(candidate, "ranking_score", candidate.edge)),
            }
            if best is None or ranked["ranking_score"] > best["ranking_score"]:
                best = ranked
        if best is not None:
            break
        if time.time() >= deadline:
            break
        time.sleep(max(args.poll_interval, 0.1))

    if best is None:
        if not available_targets:
            raise SystemExit("No next_game_winner markets found for the requested event/target game")
        raise SystemExit(
            "ML analysis did not produce a valid next_game_winner candidate for this event"
            + (f"; available target games now: {available_targets}" if available_targets else "")
        )

    candidate = best["candidate"]
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
        acceptance_probability=candidate.acceptance_probability,
        ranking_score=candidate.ranking_score,
    )

    executor = FonbetBetExecutor(dry_run=False)
    request_payloads = executor._build_request_payloads(candidate)
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
        "player1_probability": best["player1_probability"],
        "player2_probability": best["player2_probability"],
        "state_features": best["features"],
        "pre_refresh_requests": request_payloads,
    }
    if not args.skip_refresh:
        previous_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(args.refresh_timeout)
            refreshed_selection, slip_info_response = executor.refresh_candidate(candidate)
            refreshed_requests = executor._build_request_payloads(candidate)
            executor._apply_refresh_to_request_payloads(refreshed_requests, refreshed_selection)
            output["refreshed_selection"] = refreshed_selection
            output["bet_slip_info_response"] = slip_info_response
            output["ready_requests"] = refreshed_requests
            if args.send_bet:
                bet_result = executor.place_prepared_bet(candidate, refreshed_selection, slip_info_response)
                output["bet_execution"] = bet_result
        except Exception as exc:
            output["refresh_error"] = str(exc)
        finally:
            socket.setdefaulttimeout(previous_timeout)
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
