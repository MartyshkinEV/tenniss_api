try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import json
from pathlib import Path

from config import settings
from src.live import FonbetBetExecutor
from src.live.fonbet import FonbetEventDetailsClient, FonbetEventsClient, extract_fonbet_events, extract_markets_from_fonbet_catalog
from src.live.runtime import LiveMarket, ScoredSelection


def _live_root_event_rows() -> list[dict]:
    _url, payload, _headers = FonbetEventsClient().fetch_payload()
    return [
        row
        for row in extract_fonbet_events(payload)
        if int(row.get("root_sport_id") or 0) == 4
        and str(row.get("place") or "") == "live"
        and int(row.get("level") or 0) == 1
    ]


def _heuristic_candidates(market: LiveMarket) -> list[ScoredSelection]:
    candidates: list[ScoredSelection] = []
    for side, player_name, odds in (
        ("player1", market.player1_name, market.player1_odds),
        ("player2", market.player2_name, market.player2_odds),
    ):
        if odds < settings.live_min_odds or odds > settings.live_max_odds:
            continue
        implied = 1.0 / odds
        model_probability = min(implied + settings.live_edge_threshold + 0.02, 0.99)
        edge = model_probability - implied
        candidates.append(
            ScoredSelection(
                market=market,
                side=side,
                player_name=player_name,
                model_probability=model_probability,
                implied_probability=implied,
                edge=edge,
                odds=odds,
                stake=30.0,
                player_id=0,
            )
        )
    return sorted(candidates, key=lambda item: (item.edge, item.odds), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/live_betting/no_ml_probe.json")
    parser.add_argument("--place", action="store_true", help="Actually place bets for refresh-live selections.")
    args = parser.parse_args()

    details_client = FonbetEventDetailsClient()
    bet_executor = FonbetBetExecutor(dry_run=not args.place)
    report: list[dict] = []

    for event_row in _live_root_event_rows():
        root_event_id = int(event_row["event_id"])
        record = {
            "root_event_id": root_event_id,
            "match": f"{event_row.get('team1','')} vs {event_row.get('team2','')}",
            "competition": event_row.get("sport_name"),
            "status": "no_market",
        }
        try:
            event_url, payload, _headers = details_client.fetch_payload(root_event_id, 0)
            markets = [
                market
                for market in extract_markets_from_fonbet_catalog(payload)
                if market.market_type == "next_game_winner"
            ]
            record["event_url"] = event_url
            record["available_market_ids"] = [market.market_id for market in markets]
            if not markets:
                report.append(record)
                continue

            selected: ScoredSelection | None = None
            for market in markets:
                options = _heuristic_candidates(market)
                if options:
                    selected = options[0]
                    break
            if selected is None:
                record["status"] = "no_candidate"
                report.append(record)
                continue

            refreshed_selection, slip_info_response = bet_executor.refresh_candidate(selected)
            refreshed_value = refreshed_selection.get("value")
            selected_factor = (
                selected.market.raw.get("player1_factor_id")
                if selected.side == "player1"
                else selected.market.raw.get("player2_factor_id")
            )
            record.update(
                {
                    "status": "refresh_live" if refreshed_value not in (None, "", 0, 0.0) else "refresh_closed",
                    "market_id": selected.market.market_id,
                    "subevent_id": selected.market.event_id,
                    "target_game_number": selected.market.raw.get("target_game_number"),
                    "selection_side": selected.side,
                    "selection_name": selected.player_name,
                    "factor_id": selected_factor,
                    "param": (
                        selected.market.raw.get("player1_param")
                        if selected.side == "player1"
                        else selected.market.raw.get("player2_param")
                    ),
                    "odds_before_refresh": selected.odds,
                    "refreshed_selection": refreshed_selection,
                    "bet_slip_info_response": slip_info_response,
                }
            )
            if refreshed_value not in (None, "", 0, 0.0):
                result = bet_executor.place_prepared_bet(selected, refreshed_selection, slip_info_response)
                record["bet_execution"] = result
        except Exception as exc:
            record["status"] = "event_failed"
            record["error"] = str(exc)
        report.append(record)

    output = {
        "events_total": len(report),
        "refresh_live": sum(1 for item in report if item.get("status") == "refresh_live"),
        "refresh_closed": sum(1 for item in report if item.get("status") == "refresh_closed"),
        "no_candidate": sum(1 for item in report if item.get("status") == "no_candidate"),
        "no_market": sum(1 for item in report if item.get("status") == "no_market"),
        "event_failed": sum(1 for item in report if item.get("status") == "event_failed"),
        "events": report,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(output_path)
    print(
        json.dumps(
            {
                key: output[key]
                for key in ("events_total", "refresh_live", "refresh_closed", "no_candidate", "no_market", "event_failed")
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
