import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.live.fonbet import (
    DatabaseFonbetEventFeedClient,
    FonbetApiClient,
    FonbetBetExecutor,
    FonbetEventsClient,
    SnapshotRecorder,
    SnapshotMarketFeedClient,
    extract_fonbet_events,
    extract_markets_from_fonbet_catalog,
    extract_markets_from_payload,
    extract_markets_from_spoyer_payload,
)
from src.live.markov import MarkovGameModel, infer_server_side
from src.live.policy import BankrollBanditPolicy
from src.live.runtime import HistoricalLookup, LiveBettingRuntime, LiveMarket, RLDatasetLogger, RuntimeConfig, RuntimeState, ScoredSelection, build_name_aliases, select_candidate, settle_outcome_record


class StubFonbetApiClient:
    def __init__(self, slip_response, request_id_response=None, bet_response=None, bet_result_response=None):
        self.slip_response = slip_response
        self.request_id_response = request_id_response or {"result": "requestId", "requestId": 123}
        self.bet_response = bet_response or {"result": "betDelay", "betDelay": 0}
        if isinstance(bet_result_response, list):
            self.bet_result_responses = list(bet_result_response)
            self.bet_result_response = self.bet_result_responses[-1]
        else:
            self.bet_result_responses = None
            self.bet_result_response = bet_result_response or {"result": "couponResult", "coupon": {"resultCode": 0}}
        self.last_bet_payload = None

    def bet_slip_info(self, payload):
        return self.slip_response

    def bet_request_id(self, payload):
        return self.request_id_response

    def bet(self, payload):
        self.last_bet_payload = payload
        return self.bet_response

    def bet_result(self, payload):
        if self.bet_result_responses is not None:
            if len(self.bet_result_responses) > 1:
                return self.bet_result_responses.pop(0)
            return self.bet_result_responses[0]
        return self.bet_result_response


class StubMarketFeedClient:
    def __init__(self, markets):
        self.markets = markets

    def fetch_live_markets(self):
        return list(self.markets)


class StubLookup:
    def build_prediction_frame(self, market):
        row = {
            "surface": market.surface,
            "tourney_level": market.tourney_level,
            "round": market.round_name,
            "best_of": market.best_of,
            "p1_hold_rate": 0.70,
            "p2_hold_rate": 0.62,
            "p1_break_rate": 0.31,
            "p2_break_rate": 0.28,
        }
        return pd.DataFrame([row]), row, 1, 2


class StubBetExecutor:
    def refresh_candidate(self, candidate):
        return {
            "score": candidate.market.raw.get("score"),
            "value": candidate.market.raw.get("player1_value"),
            "param": candidate.market.raw.get("player1_param"),
            "factor_id": candidate.market.raw.get("player1_factor_id"),
        }, {"result": "betSlipInfo", "bets": []}

    def place_prepared_bet(self, candidate, refreshed_selection, slip_info_response):
        return {
            "status": "dry_run",
            "request": {"event_id": candidate.market.event_id},
            "refreshed_selection": refreshed_selection,
            "bet_slip_info_response": slip_info_response,
        }

    def place_express_bet(self, candidates, stake=None):
        return {
            "status": "dry_run",
            "request": {
                "event_ids": [candidate.market.event_id for candidate in candidates],
                "stake": stake,
            },
        }


class StubRefreshDeadBetExecutor:
    def refresh_candidate(self, candidate):
        return {
            "score": candidate.market.raw.get("score"),
            "value": 0.0,
            "param": candidate.market.raw.get("player2_param"),
            "factor_id": candidate.market.raw.get("player2_factor_id"),
        }, {"result": "betSlipInfo", "bets": [{"factor": {"v": 0}}]}

    def place_prepared_bet(self, candidate, refreshed_selection, slip_info_response):
        return {
            "status": "dry_run",
            "request": {"event_id": candidate.market.event_id, "selection": candidate.side},
            "refreshed_selection": refreshed_selection,
            "bet_slip_info_response": slip_info_response,
        }


class LiveRuntimeTest(unittest.TestCase):
    def test_fonbet_bet_executor_marks_nonzero_coupon_result_as_not_placed(self):
        executor = FonbetBetExecutor(dry_run=True)
        self.assertEqual(executor._placement_status({"result": "couponResult", "coupon": {"resultCode": 0}}), "placed")
        self.assertEqual(executor._placement_status({"result": "couponResult", "coupon": {"resultCode": 2}}), "odds_changed")
        self.assertEqual(executor._placement_status({"result": "couponResult", "coupon": {"resultCode": 100}}), "temporarily_suspended")

    def test_historical_lookup_uses_player_csv_and_neutral_stats_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            player_csv = tmp / "atp_players.csv"
            player_csv.write_text(
                "\n".join(
                    [
                        "player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id",
                        "123456,Marco,Rossi,R,20000101,ITA,185,Q1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            player_match_stats = pd.DataFrame(
                columns=[
                    "match_id",
                    "surface",
                    "tourney_date",
                    "player_id",
                    "player_name",
                    "opponent_id",
                    "opponent_name",
                    "is_win",
                    "player_rank",
                    "player_rank_points",
                    "minutes",
                    "service_games",
                    "bp_faced",
                    "bp_saved",
                    "opp_bp_faced",
                    "opp_bp_saved",
                    "opp_service_games",
                ]
            )
            match_features_elo = pd.DataFrame(columns=["p1_id", "p2_id", "tourney_date", "match_id"])
            from unittest.mock import patch

            with patch("src.live.runtime.discover_player_csv_files", return_value=[player_csv]):
                lookup = HistoricalLookup(
                    player_match_stats=player_match_stats,
                    match_features_elo=match_features_elo,
                )

            self.assertEqual(lookup.resolve_player_id("Rossi M"), 123456)
            stats = lookup.player_stats(123456, "Hard")
            self.assertEqual(stats["winrate_last10"], 0.5)
            self.assertEqual(stats["hold_rate"], 0.62)

    def test_extract_markets_from_payload_filters_tennis(self):
        payload = {
            "events": [
                {
                    "id": 101,
                    "sport": "Tennis",
                    "player1": "Roger Federer",
                    "player2": "Rafael Nadal",
                    "player1_odds": 2.1,
                    "player2_odds": 1.8,
                    "surface": "Clay",
                    "round": "R16",
                },
                {
                    "id": 102,
                    "sport": "Football",
                    "player1": "A",
                    "player2": "B",
                    "player1_odds": 2.1,
                    "player2_odds": 1.8,
                },
            ]
        }

        markets = extract_markets_from_payload(payload)
        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0].market_id, "101")
        self.assertEqual(markets[0].surface, "Clay")

    def test_extract_markets_from_spoyer_payload(self):
        payload = {
            "games_live": [
                {
                    "game_id": "201",
                    "home": {"name": "Carlos Alcaraz"},
                    "away": {"name": "Jannik Sinner"},
                    "league": {"name": "ATP Finals"},
                    "surface": "Hard",
                    "home_od": {"value": "1.91"},
                    "away_od": {"value": "1.95"},
                    "round": "Semi-final",
                    "best_of": 3,
                }
            ]
        }

        markets = extract_markets_from_spoyer_payload(payload)
        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0].market_id, "201")
        self.assertEqual(markets[0].competition, "ATP Finals")
        self.assertEqual(markets[0].player1_odds, 1.91)

    def test_extract_markets_from_fonbet_catalog(self):
        payload = {
            "sports": [
                {"id": 4, "kind": "sport", "name": "Tennis", "alias": "tennis"},
                {"id": 71736, "parentId": 4, "kind": "segment", "name": "Open tournament Liga Pro. Men. Balashikha"},
            ],
            "events": [
                {
                    "id": 63203793,
                    "level": 1,
                    "sportId": 71736,
                    "place": "live",
                    "team1": "Gorbachyov Aleksandr",
                    "team2": "Parypchik Dmitrii",
                }
            ],
            "eventMiscs": [
                {"id": 63203793, "score1": 1, "score2": 0, "liveDelay": 0, "comment": "(6-4 0-5)"}
            ],
            "customFactors": [
                {"e": 63203793, "factors": [{"f": 921, "v": 2.55}, {"f": 923, "v": 1.46}]}
            ],
        }

        markets = extract_markets_from_fonbet_catalog(payload)
        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0].event_id, "63203793")
        self.assertEqual(markets[0].player1_odds, 2.55)
        self.assertEqual(markets[0].player2_odds, 1.46)
        self.assertEqual(markets[0].raw["player2_factor_id"], 923)
        self.assertEqual(markets[0].raw["score"], "1:0")
        self.assertEqual(markets[0].market_type, "match_winner")

    def test_extract_fonbet_events_keeps_all_events_and_enriches_sport_fields(self):
        payload = {
            "sports": [
                {"id": 4, "kind": "sport", "name": "Tennis", "alias": "tennis"},
                {"id": 71736, "parentId": 4, "kind": "segment", "name": "Open tournament Liga Pro. Men. Balashikha", "alias": "liga-pro-men"},
            ],
            "events": [
                {
                    "id": 63203793,
                    "level": 1,
                    "sportId": 71736,
                    "place": "live",
                    "team1": "Gorbachyov Aleksandr",
                    "team2": "Parypchik Dmitrii",
                },
                {
                    "id": 63203794,
                    "parentId": 63203793,
                    "level": 2,
                    "sportId": 71736,
                    "place": "live",
                    "name": "2nd set",
                },
            ],
        }

        events = extract_fonbet_events(payload)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event_id"], 63203793)
        self.assertEqual(events[0]["sport_id"], 71736)
        self.assertEqual(events[0]["sport_name"], "Open tournament Liga Pro. Men. Balashikha")
        self.assertEqual(events[0]["sport_alias"], "liga-pro-men")
        self.assertEqual(events[0]["root_sport_id"], 4)
        self.assertEqual(events[0]["root_sport_name"], "Tennis")
        self.assertEqual(events[1]["parent_id"], 63203793)
        self.assertEqual(events[1]["event_name"], "2nd set")

    def test_extract_next_game_market_from_fonbet_catalog(self):
        payload = {
            "sports": [
                {"id": 4, "kind": "sport", "name": "Tennis", "alias": "tennis"},
                {"id": 105476, "parentId": 4, "kind": "segment", "name": "ITF. Women. Kofu. Qualification"},
            ],
            "events": [
                {
                    "id": 63280940,
                    "level": 1,
                    "sportId": 105476,
                    "place": "live",
                    "team1": "Okamura K",
                    "team2": "Ozeki M",
                },
                {
                    "id": 63304562,
                    "parentId": 63280940,
                    "name": "2nd set",
                    "level": 2,
                    "sportId": 105476,
                    "place": "live",
                }
            ],
            "eventMiscs": [
                {"id": 63304562, "score1": 0, "score2": 1, "liveDelay": 0, "comment": "(00*-30)", "servingTeam": 1}
            ],
            "customFactors": [
                {
                    "e": 63304562,
                    "factors": [
                        {"f": 1750, "v": 1.37, "p": 200},
                        {"f": 1751, "v": 2.95, "p": 200},
                        {"f": 1753, "v": 1.62, "p": 400},
                        {"f": 1754, "v": 2.15, "p": 400},
                    ],
                }
            ],
        }

        markets = extract_markets_from_fonbet_catalog(payload)
        game_markets = [market for market in markets if market.market_type == "next_game_winner"]
        self.assertEqual(len(game_markets), 2)
        self.assertEqual(game_markets[0].player1_name, "Okamura K")
        self.assertEqual(game_markets[0].player2_name, "Ozeki M")
        self.assertEqual(game_markets[0].raw["player1_factor_id"], 1750)
        self.assertEqual(game_markets[0].raw["player1_param"], 200)
        self.assertEqual(game_markets[0].raw["target_game_number"], 2)
        self.assertEqual(game_markets[0].raw["zone"], "es")
        self.assertEqual(game_markets[1].raw["player1_factor_id"], 1753)
        self.assertEqual(game_markets[1].raw["player1_param"], 400)
        self.assertEqual(game_markets[1].raw["target_game_number"], 4)

    def test_extract_next_game_market_keeps_future_games_even_if_offset_is_higher(self):
        payload = {
            "sports": [
                {"id": 4, "kind": "sport", "name": "Tennis", "alias": "tennis"},
                {"id": 62250, "parentId": 4, "kind": "segment", "name": "ATP Challenger. Zadar. Qualification"},
            ],
            "events": [
                {
                    "id": 63295680,
                    "level": 1,
                    "sportId": 62250,
                    "place": "live",
                    "team1": "Sperle J",
                    "team2": "Popovic S",
                },
                {
                    "id": 63316465,
                    "parentId": 63295680,
                    "name": "2nd set",
                    "level": 2,
                    "sportId": 62250,
                    "place": "live",
                },
            ],
            "eventMiscs": [
                {"id": 63316465, "score1": 4, "score2": 5, "liveDelay": 3, "comment": "(15*-15)", "servingTeam": 1}
            ],
            "customFactors": [
                {
                    "e": 63316465,
                    "factors": [
                        {"f": 1750, "v": 1.35, "p": 1000, "pt": "10"},
                        {"f": 1751, "v": 2.8, "p": 1000, "pt": "10"},
                        {"f": 1753, "v": 2.9, "p": 1100, "pt": "11"},
                        {"f": 1754, "v": 1.35, "p": 1100, "pt": "11"},
                    ],
                }
            ],
        }

        markets = extract_markets_from_fonbet_catalog(payload)
        game_markets = [market for market in markets if market.market_type == "next_game_winner"]
        self.assertEqual(len(game_markets), 2)
        self.assertEqual([market.raw["target_game_number"] for market in game_markets], [10, 11])

    def test_extract_next_game_market_includes_first_game_factor_pair(self):
        payload = {
            "sports": [
                {"id": 4, "kind": "sport", "name": "Tennis", "alias": "tennis"},
                {"id": 62250, "parentId": 4, "kind": "segment", "name": "ATP Challenger. Zadar. Qualification"},
            ],
            "events": [
                {
                    "id": 63312750,
                    "level": 1,
                    "sportId": 62250,
                    "place": "live",
                    "team1": "Pieri S",
                    "team2": "Campana Lee G",
                },
                {
                    "id": 63337687,
                    "parentId": 63312750,
                    "name": "3rd set",
                    "level": 2,
                    "sportId": 62250,
                    "place": "live",
                },
            ],
            "eventMiscs": [
                {"id": 63337687, "score1": 0, "score2": 0, "liveDelay": 3, "comment": "(00-00*)", "servingTeam": 2}
            ],
            "customFactors": [
                {
                    "e": 63337687,
                    "factors": [
                        {"f": 1747, "v": 3.30, "p": 100, "pt": "1"},
                        {"f": 1748, "v": 1.27, "p": 100, "pt": "1"},
                        {"f": 1750, "v": 1.30, "p": 200, "pt": "2"},
                        {"f": 1751, "v": 3.15, "p": 200, "pt": "2"},
                    ],
                }
            ],
        }

        markets = extract_markets_from_fonbet_catalog(payload)
        game_markets = [market for market in markets if market.market_type == "next_game_winner"]
        self.assertEqual(len(game_markets), 1)
        self.assertEqual(game_markets[0].player1_name, "Pieri S")
        self.assertEqual(game_markets[0].player2_name, "Campana Lee G")
        self.assertEqual(game_markets[0].raw["player1_factor_id"], 1747)
        self.assertEqual(game_markets[0].raw["player2_factor_id"], 1748)
        self.assertEqual(game_markets[0].raw["target_game_number"], 1)

    def test_extract_point_plus_one_market_from_fonbet_catalog(self):
        payload = {
            "sports": [
                {"id": 4, "kind": "sport", "name": "Tennis", "alias": "tennis"},
                {"id": 122604, "parentId": 4, "kind": "segment", "name": "WTA 125K. Antalya 3"},
            ],
            "events": [
                {
                    "id": 63278962,
                    "level": 1,
                    "sportId": 122604,
                    "place": "live",
                    "team1": "Kalinina A",
                    "team2": "Zidansek T",
                },
                {
                    "id": 63308713,
                    "parentId": 63278962,
                    "name": "2nd set",
                    "level": 2,
                    "sportId": 122604,
                    "place": "live",
                },
            ],
            "eventMiscs": [
                {"id": 63308713, "score1": 4, "score2": 3, "liveDelay": 0, "comment": "(40-30*)", "servingTeam": 2}
            ],
            "customFactors": [
                {
                    "e": 63308713,
                    "factors": [
                        {"f": 2998, "v": 1.8, "p": 524290, "lo": 8, "hi": 8, "pt": "8 - point 2"},
                        {"f": 2999, "v": 1.95, "p": 524290, "lo": 8, "hi": 8, "pt": "8 - point 2"},
                    ],
                }
            ],
        }
        markets = extract_markets_from_fonbet_catalog(payload)
        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0].market_type, "point_plus_one_winner")
        self.assertEqual(markets[0].raw["player1_factor_id"], 2998)
        self.assertEqual(markets[0].raw["target_point_number"], 8)
        self.assertEqual(markets[0].raw["player1_param"], 524290)

    def test_select_candidate_uses_edge_threshold(self):
        market = LiveMarket(
            market_id="m1",
            event_id="e1",
            competition="ATP",
            surface="Hard",
            round_name="R32",
            best_of=3,
            tourney_level="tour",
            player1_name="Player One",
            player2_name="Player Two",
            player1_odds=2.4,
            player2_odds=1.6,
            raw={},
        )
        config = RuntimeConfig(
            model_path=Path("models/match_winner/lightgbm_elo.joblib"),
            poll_interval_seconds=20,
            edge_threshold=0.05,
            min_model_probability=0.55,
            min_odds=1.5,
            max_odds=3.5,
            default_stake=100.0,
            bankroll=0.0,
            kelly_fraction=0.25,
            dry_run=True,
            state_path=Path("state.json"),
            decisions_path=Path("decisions.jsonl"),
            rl_snapshots_path=Path("rl_snapshots.jsonl"),
            rl_actions_path=Path("rl_actions.jsonl"),
            rl_outcomes_path=Path("rl_outcomes.jsonl"),
            rl_tracker_state_path=Path("rl_tracker_state.json"),
            rl_market_close_cycles=3,
        )

        candidate = select_candidate(
            market=market,
            player1_probability=0.62,
            player2_probability=0.38,
            player1_id=1,
            player2_id=2,
            config=config,
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.side, "player1")
        self.assertEqual(candidate.stake, 100.0)
        self.assertGreater(candidate.edge, 0.05)

    def test_build_name_aliases_supports_initial_forms(self):
        aliases = build_name_aliases("rafael nadal")
        self.assertIn("rafael nadal", aliases)
        self.assertIn("nadal r", aliases)
        self.assertIn("r nadal", aliases)

    def test_runtime_state_persists_placed_selection_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "state.json"
            state = RuntimeState(path)
            self.assertFalse(state.has_seen("m1:player1"))
            state.mark_placed("m1:player1")

            reloaded = RuntimeState(path)
            self.assertTrue(reloaded.has_seen("m1:player1"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["placed_selection_ids"], ["m1:player1"])
            self.assertEqual(payload["current_bankroll"], 0.0)

    def test_rl_dataset_logger_appends_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshots = Path(tmp_dir) / "rl_snapshots.jsonl"
            actions = Path(tmp_dir) / "rl_actions.jsonl"
            outcomes = Path(tmp_dir) / "rl_outcomes.jsonl"
            point_trajectories = Path(tmp_dir) / "point_trajectories.jsonl"
            logger = RLDatasetLogger(snapshots, actions, outcomes, point_trajectories)

            logger.log_snapshot({"event_id": "e1", "status": "no_edge"})
            logger.log_action({"event_id": "e1", "action": "no_bet"})
            logger.log_outcome({"event_id": "e1", "status": "pending"})
            logger.log_point_trajectory({"event_id": "e1", "stage": "observed"})

            snapshot_lines = snapshots.read_text(encoding="utf-8").strip().splitlines()
            action_lines = actions.read_text(encoding="utf-8").strip().splitlines()
            outcome_lines = outcomes.read_text(encoding="utf-8").strip().splitlines()
            point_lines = point_trajectories.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(snapshot_lines), 1)
            self.assertEqual(len(action_lines), 1)
            self.assertEqual(len(outcome_lines), 1)
            self.assertEqual(len(point_lines), 1)
            self.assertEqual(json.loads(snapshot_lines[0])["status"], "no_edge")
            self.assertEqual(json.loads(action_lines[0])["action"], "no_bet")
            self.assertEqual(json.loads(outcome_lines[0])["status"], "pending")
            self.assertEqual(json.loads(point_lines[0])["stage"], "observed")

    def test_snapshot_market_feed_client_reads_latest_market_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rl_snapshots.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event_id": "e1",
                                "market_id": "m1",
                                "competition": "ITF",
                                "surface": "Hard",
                                "round_name": "2nd set",
                                "best_of": 3,
                                "tourney_level": "itf",
                                "player1_name": "Player One",
                                "player2_name": "Player Two",
                                "player1_odds": 1.9,
                                "player2_odds": 1.9,
                                "market_type": "point_plus_one_winner",
                                "live_score": "4:3",
                                "live_comment": "(40-30*)",
                                "live_delay": 0,
                                "serving_team": 2,
                                "player1_factor_id": 2998,
                                "player2_factor_id": 2999,
                                "player1_param": 524290,
                                "player2_param": 524290,
                                "scope_market_id": "1600",
                                "target_point_number": 8,
                            }
                        ),
                        json.dumps(
                            {
                                "event_id": "e1",
                                "market_id": "m1",
                                "competition": "ITF",
                                "surface": "Hard",
                                "round_name": "2nd set",
                                "best_of": 3,
                                "tourney_level": "itf",
                                "player1_name": "Player One",
                                "player2_name": "Player Two",
                                "player1_odds": 1.8,
                                "player2_odds": 2.0,
                                "market_type": "point_plus_one_winner",
                                "live_score": "5:3",
                                "live_comment": "(40-15*)",
                                "live_delay": 0,
                                "serving_team": 2,
                                "player1_factor_id": 2998,
                                "player2_factor_id": 2999,
                                "player1_param": 589826,
                                "player2_param": 589826,
                                "scope_market_id": "1600",
                                "target_point_number": 9,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            markets = SnapshotMarketFeedClient(path).fetch_live_markets()
            self.assertEqual(len(markets), 1)
            self.assertEqual(markets[0].market_id, "m1")
            self.assertEqual(markets[0].player1_odds, 1.8)
            self.assertEqual(markets[0].raw["player1_factor_id"], 2998)
            self.assertEqual(markets[0].raw["target_point_number"], 9)

    def test_snapshot_recorder_writes_normalized_markets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "market_snapshots.jsonl"
            recorder = SnapshotRecorder(path)
            markets = [
                LiveMarket(
                    market_id="m1",
                    event_id="e1",
                    competition="ITF",
                    surface="Hard",
                    round_name="2nd set",
                    best_of=3,
                    tourney_level="itf",
                    player1_name="Player One",
                    player2_name="Player Two",
                    player1_odds=1.8,
                    player2_odds=2.0,
                    market_type="point_plus_one_winner",
                    raw={
                        "timestamp_utc": "2026-03-15T10:00:00+00:00",
                        "score": "5:3",
                        "comment": "(40-15*)",
                        "liveDelay": 0,
                        "serveT": 2,
                        "player1_factor_id": 2998,
                        "player2_factor_id": 2999,
                        "player1_param": 589826,
                        "player2_param": 589826,
                        "scopeMarketId": "1600",
                        "target_point_number": 9,
                        "zone": "es",
                    },
                )
            ]
            written = recorder.write_markets(markets)
            self.assertEqual(written, 1)
            payload = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["market_id"], "m1")
            self.assertEqual(payload["target_point_number"], 9)
            self.assertEqual(payload["player1_factor_id"], 2998)

    def test_runtime_logs_point_trajectory_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            market = LiveMarket(
                market_id="63308713:point:8",
                event_id="63308713",
                competition="WTA 125K. Antalya 3",
                surface="Hard",
                round_name="2nd set",
                best_of=3,
                tourney_level="tour",
                player1_name="Kalinina A",
                player2_name="Zidansek T",
                player1_odds=1.8,
                player2_odds=1.95,
                market_type="point_plus_one_winner",
                raw={
                    "score": "4:3",
                    "comment": "(40-30*)",
                    "serveT": 2,
                    "scopeMarketId": "1600",
                    "player1_factor_id": 2998,
                    "player2_factor_id": 2999,
                    "player1_value": 1.8,
                    "player2_value": 1.95,
                    "player1_param": 524290,
                    "player2_param": 524290,
                    "param": 524290,
                    "target_point_number": 8,
                    "zone": "es",
                },
            )
            config = RuntimeConfig(
                model_path=tmp / "dummy_model.joblib",
                poll_interval_seconds=20,
                edge_threshold=0.01,
                min_model_probability=0.5,
                min_odds=1.2,
                max_odds=3.5,
                default_stake=30.0,
                bankroll=0.0,
                kelly_fraction=0.25,
                dry_run=True,
                state_path=tmp / "state.json",
                decisions_path=tmp / "decisions.jsonl",
                rl_snapshots_path=tmp / "rl_snapshots.jsonl",
                rl_actions_path=tmp / "rl_actions.jsonl",
                rl_outcomes_path=tmp / "rl_outcomes.jsonl",
                rl_tracker_state_path=tmp / "rl_tracker_state.json",
                rl_market_close_cycles=3,
                point_trajectories_path=tmp / "point_trajectories.jsonl",
            )
            joblib.dump(object(), config.model_path)
            runtime = LiveBettingRuntime(
                market_feed_client=StubMarketFeedClient([market]),
                bet_executor=StubBetExecutor(),
                config=config,
                lookup=StubLookup(),
            )

            runtime.run_cycle()

            point_lines = (tmp / "point_trajectories.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(point_lines), 2)
            stages = {json.loads(line)["stage"] for line in point_lines}
            self.assertIn("scored", stages)
            self.assertIn("bet_dry_run", stages)

    def test_runtime_uses_point_fast_mode_when_refresh_kills_market(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            market = LiveMarket(
                market_id="63309103:point:7",
                event_id="63309103",
                competition="ITF. Men. India",
                surface="Hard",
                round_name="2nd set",
                best_of=3,
                tourney_level="itf",
                player1_name="Sekulic P",
                player2_name="Singh Karan",
                player1_odds=1.95,
                player2_odds=1.75,
                market_type="point_plus_one_winner",
                raw={
                    "score": "2:4",
                    "comment": "(40-15*)",
                    "serveT": 2,
                    "scopeMarketId": "1600",
                    "player1_factor_id": 2995,
                    "player2_factor_id": 2996,
                    "player1_value": 1.95,
                    "player2_value": 1.75,
                    "player1_param": 458759,
                    "player2_param": 458759,
                    "param": 458759,
                    "target_point_number": 7,
                    "zone": "es",
                },
            )
            config = RuntimeConfig(
                model_path=tmp / "dummy_model.joblib",
                poll_interval_seconds=20,
                edge_threshold=0.01,
                min_model_probability=0.5,
                min_odds=1.2,
                max_odds=3.5,
                default_stake=30.0,
                bankroll=0.0,
                kelly_fraction=0.25,
                dry_run=True,
                state_path=tmp / "state.json",
                decisions_path=tmp / "decisions.jsonl",
                rl_snapshots_path=tmp / "rl_snapshots.jsonl",
                rl_actions_path=tmp / "rl_actions.jsonl",
                rl_outcomes_path=tmp / "rl_outcomes.jsonl",
                rl_tracker_state_path=tmp / "rl_tracker_state.json",
                rl_market_close_cycles=3,
                point_trajectories_path=tmp / "point_trajectories.jsonl",
                point_fast_mode=True,
            )
            joblib.dump(object(), config.model_path)
            runtime = LiveBettingRuntime(
                market_feed_client=StubMarketFeedClient([market]),
                bet_executor=StubRefreshDeadBetExecutor(),
                config=config,
                lookup=StubLookup(),
            )

            actions = runtime.run_cycle()

            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0]["mode"], "point_fast_mode")
            point_lines = (tmp / "point_trajectories.jsonl").read_text(encoding="utf-8").strip().splitlines()
            stages = {json.loads(line)["stage"] for line in point_lines}
            self.assertIn("point_fast_mode_fallback", stages)
            self.assertIn("bet_dry_run_fast_mode", stages)

    def test_runtime_places_express_from_top_distinct_events(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            config = RuntimeConfig(
                model_path=tmp / "dummy_model.joblib",
                poll_interval_seconds=20,
                edge_threshold=0.01,
                min_model_probability=0.5,
                min_odds=1.2,
                max_odds=3.5,
                default_stake=30.0,
                bankroll=0.0,
                kelly_fraction=0.25,
                dry_run=True,
                state_path=tmp / "state.json",
                decisions_path=tmp / "decisions.jsonl",
                rl_snapshots_path=tmp / "rl_snapshots.jsonl",
                rl_actions_path=tmp / "rl_actions.jsonl",
                rl_outcomes_path=tmp / "rl_outcomes.jsonl",
                rl_tracker_state_path=tmp / "rl_tracker_state.json",
                rl_market_close_cycles=3,
                point_trajectories_path=tmp / "point_trajectories.jsonl",
                bet_mode="express",
                express_size=2,
            )
            joblib.dump(object(), config.model_path)
            markets = [
                LiveMarket(
                    market_id="m1",
                    event_id="e1",
                    competition="ATP",
                    surface="Hard",
                    round_name="R32",
                    best_of=3,
                    tourney_level="tour",
                    player1_name="A",
                    player2_name="B",
                    player1_odds=2.1,
                    player2_odds=1.8,
                ),
                LiveMarket(
                    market_id="m2",
                    event_id="e2",
                    competition="ATP",
                    surface="Clay",
                    round_name="R16",
                    best_of=3,
                    tourney_level="tour",
                    player1_name="C",
                    player2_name="D",
                    player1_odds=1.9,
                    player2_odds=1.9,
                ),
                LiveMarket(
                    market_id="m3",
                    event_id="e1",
                    competition="ATP",
                    surface="Grass",
                    round_name="QF",
                    best_of=3,
                    tourney_level="tour",
                    player1_name="E",
                    player2_name="F",
                    player1_odds=2.2,
                    player2_odds=1.7,
                ),
            ]
            runtime = LiveBettingRuntime(
                market_feed_client=StubMarketFeedClient(markets),
                bet_executor=StubBetExecutor(),
                config=config,
                lookup=StubLookup(),
            )
            scored_candidates = [
                ScoredSelection(markets[0], "player1", "A", 0.62, 1 / 2.1, 0.14, 2.1, 30.0, 1),
                ScoredSelection(markets[1], "player1", "C", 0.60, 1 / 1.9, 0.10, 1.9, 30.0, 3),
                ScoredSelection(markets[2], "player1", "E", 0.67, 1 / 2.2, 0.22, 2.2, 30.0, 5),
            ]
            refresh_payloads = [
                (markets[0], {"value": 2.1}, {"result": "betSlipInfo"}, {"surface": "Hard"}, 0.62, 0.38, scored_candidates[0]),
                (markets[1], {"value": 1.9}, {"result": "betSlipInfo"}, {"surface": "Clay"}, 0.60, 0.40, scored_candidates[1]),
                (markets[2], {"value": 2.2}, {"result": "betSlipInfo"}, {"surface": "Grass"}, 0.67, 0.33, scored_candidates[2]),
            ]

            from unittest.mock import MagicMock

            runtime._score_market_details = MagicMock(
                side_effect=[
                    (pd.DataFrame([{"surface": "Hard"}]), {"surface": "Hard"}, 0.62, 0.38, scored_candidates[0]),
                    (pd.DataFrame([{"surface": "Clay"}]), {"surface": "Clay"}, 0.60, 0.40, scored_candidates[1]),
                    (pd.DataFrame([{"surface": "Grass"}]), {"surface": "Grass"}, 0.67, 0.33, scored_candidates[2]),
                ]
            )
            runtime._refresh_and_rescore = MagicMock(side_effect=refresh_payloads)
            runtime.bet_executor.place_express_bet = MagicMock(
                return_value={"status": "dry_run", "bet_result_response": {"coupon": {}}}
            )

            actions = runtime.run_cycle()

            placed_candidates = runtime.bet_executor.place_express_bet.call_args.args[0]
            self.assertEqual([candidate.market.event_id for candidate in placed_candidates], ["e1", "e2"])
            self.assertEqual(placed_candidates[0].player_name, "E")
            self.assertEqual(placed_candidates[1].player_name, "C")
            self.assertEqual(len(actions), 2)
            self.assertTrue(all(action["mode"] == "express" for action in actions))

    def test_fonbet_api_client_builds_browser_like_headers(self):
        client = FonbetApiClient(cookie="sid=test", auth_token="abc")
        headers = client._headers()
        self.assertEqual(headers["Content-Type"], "text/plain;charset=UTF-8")
        self.assertEqual(headers["Origin"], "https://fon.bet")
        self.assertEqual(headers["Cookie"], "sid=test")
        self.assertEqual(headers["Authorization"], "Bearer abc")

    def test_fonbet_events_client_builds_catalog_url_from_host(self):
        client = FonbetEventsClient(feed_url="https://line-lb61-w.bk6bba-resources.com/ma/line/liveEvents")
        self.assertEqual(
            client._catalog_url(),
            "https://line-lb61-w.bk6bba-resources.com/ma/events/list?lang=en&scopeMarket=1600",
        )

    def test_database_fonbet_event_feed_client_extracts_markets_from_latest_snapshot(self):
        class StubResult:
            def mappings(self):
                return self

            def first(self):
                return {
                    "payload_json": {
                        "sports": [
                            {"id": 4, "kind": "sport", "name": "Tennis", "alias": "tennis"},
                            {"id": 71736, "parentId": 4, "kind": "segment", "name": "Open tournament Liga Pro. Men. Balashikha"},
                        ],
                        "events": [
                            {
                                "id": 63203793,
                                "level": 1,
                                "sportId": 71736,
                                "place": "live",
                                "team1": "Gorbachyov Aleksandr",
                                "team2": "Parypchik Dmitrii",
                            }
                        ],
                        "eventMiscs": [
                            {"id": 63203793, "score1": 1, "score2": 0, "liveDelay": 0, "comment": "(6-4 0-5)"}
                        ],
                        "customFactors": [
                            {"e": 63203793, "factors": [{"f": 921, "v": 2.55}, {"f": 923, "v": 1.46}]}
                        ],
                    }
                }

        class StubConnection:
            def execute(self, _query):
                return StubResult()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class StubEngine:
            def connect(self):
                return StubConnection()

        client = DatabaseFonbetEventFeedClient(engine=StubEngine())
        markets = client.fetch_live_markets()
        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0].event_id, "63203793")

    def test_fonbet_session_payload_shape(self):
        executor = FonbetBetExecutor(dry_run=True)
        payload = executor._session_payload_defaults()
        self.assertEqual(payload["lang"], "en")
        self.assertEqual(payload["sysId"], 21)
        self.assertEqual(payload["CDI"], 0)

    def test_fonbet_executor_uses_payload_templates_in_dry_run(self):
        market = LiveMarket(
            market_id="m1",
            event_id="12345",
            competition="ATP",
            surface="Hard",
            round_name="R32",
            best_of=3,
            tourney_level="tour",
            player1_name="Player One",
            player2_name="Player Two",
            player1_odds=2.4,
            player2_odds=1.6,
            raw={
                "factor": 921,
                "score": "0:1",
                "scopeMarketId": "1600",
                "fonbet_payloads": {
                    "bet_slip_info": {"fsid": "session1"},
                    "bet_request_id": {"deviceId": "device1"},
                    "bet": {"coupon": {"system": 0}, "fsid": "session1"},
                    "bet_result": {"requestId": "req-1", "fsid": "session1"},
                }
            },
        )
        candidate = select_candidate(
            market=market,
            player1_probability=0.62,
            player2_probability=0.38,
            player1_id=1,
            player2_id=2,
            config=RuntimeConfig(
                model_path=Path("models/match_winner/lightgbm_elo.joblib"),
                poll_interval_seconds=20,
                edge_threshold=0.05,
                min_model_probability=0.55,
                min_odds=1.5,
                max_odds=3.5,
                default_stake=100.0,
                bankroll=0.0,
                kelly_fraction=0.25,
                dry_run=True,
                state_path=Path("state.json"),
                decisions_path=Path("decisions.jsonl"),
                rl_snapshots_path=Path("rl_snapshots.jsonl"),
                rl_actions_path=Path("rl_actions.jsonl"),
                rl_outcomes_path=Path("rl_outcomes.jsonl"),
                rl_tracker_state_path=Path("rl_tracker_state.json"),
                rl_market_close_cycles=3,
            ),
        )

        executor = FonbetBetExecutor(dry_run=True)
        result = executor.place_bet(candidate)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["fonbet_requests"]["bet_slip_info"]["fsid"], "session1")
        self.assertEqual(result["fonbet_requests"]["bet_slip_info"]["scopeMarketId"], "1600")
        self.assertEqual(result["fonbet_requests"]["bet_slip_info"]["bets"][0]["eventId"], 12345)
        self.assertEqual(result["fonbet_requests"]["bet_request_id"]["deviceId"], "device1")
        self.assertEqual(result["fonbet_requests"]["bet"]["coupon"]["system"], 0)
        self.assertEqual(result["fonbet_requests"]["bet"]["coupon"]["amount"], 100.0)
        self.assertEqual(result["fonbet_requests"]["bet"]["coupon"]["bets"][0]["zone"], "lv")
        self.assertEqual(result["fonbet_requests"]["bet"]["coupon"]["bets"][0]["factor"], 921)
        self.assertEqual(result["fonbet_requests"]["bet_result"]["requestId"], "req-1")

    def test_fonbet_executor_express_coerces_total_param_to_number(self):
        markets = [
            LiveMarket(
                market_id="m1",
                event_id="1001",
                competition="ATP",
                surface="Hard",
                round_name="1st set",
                best_of=3,
                tourney_level="tour",
                player1_name="Player One",
                player2_name="Player Two",
                player1_odds=2.1,
                player2_odds=1.7,
                market_type="set_total_over_under",
                raw={"player1_factor_id": 1848, "player1_param": "950", "score": "1:1", "zone": "lv"},
            ),
            LiveMarket(
                market_id="m2",
                event_id="1002",
                competition="ATP",
                surface="Hard",
                round_name="1st set",
                best_of=3,
                tourney_level="tour",
                player1_name="Player Three",
                player2_name="Player Four",
                player1_odds=2.2,
                player2_odds=1.6,
                market_type="set_total_over_under",
                raw={"player1_factor_id": 1848, "player1_param": "1050", "score": "2:1", "zone": "lv"},
            ),
        ]
        candidates = [
            select_candidate(
                market=LiveMarket(
                    market_id=markets[0].market_id,
                    event_id=markets[0].event_id,
                    competition=markets[0].competition,
                    surface=markets[0].surface,
                    round_name=markets[0].round_name,
                    best_of=markets[0].best_of,
                    tourney_level=markets[0].tourney_level,
                    player1_name=markets[0].player1_name,
                    player2_name=markets[0].player2_name,
                    player1_odds=markets[0].player1_odds,
                    player2_odds=markets[0].player2_odds,
                    market_type=markets[0].market_type,
                    raw=markets[0].raw,
                ),
                player1_probability=0.7,
                player2_probability=0.3,
                player1_id=0,
                player2_id=0,
                config=RuntimeConfig(
                    model_path=Path("models/match_winner/lightgbm_elo.joblib"),
                    poll_interval_seconds=20,
                    edge_threshold=0.05,
                    min_model_probability=0.55,
                    min_odds=1.5,
                    max_odds=3.5,
                    default_stake=30.0,
                    bankroll=0.0,
                    kelly_fraction=0.25,
                    dry_run=True,
                    state_path=Path("state.json"),
                    decisions_path=Path("decisions.jsonl"),
                    rl_snapshots_path=Path("rl_snapshots.jsonl"),
                    rl_actions_path=Path("rl_actions.jsonl"),
                    rl_outcomes_path=Path("rl_outcomes.jsonl"),
                    rl_tracker_state_path=Path("rl_tracker_state.json"),
                    rl_market_close_cycles=3,
                ),
            ),
            select_candidate(
                market=LiveMarket(
                    market_id=markets[1].market_id,
                    event_id=markets[1].event_id,
                    competition=markets[1].competition,
                    surface=markets[1].surface,
                    round_name=markets[1].round_name,
                    best_of=markets[1].best_of,
                    tourney_level=markets[1].tourney_level,
                    player1_name=markets[1].player1_name,
                    player2_name=markets[1].player2_name,
                    player1_odds=markets[1].player1_odds,
                    player2_odds=markets[1].player2_odds,
                    market_type=markets[1].market_type,
                    raw=markets[1].raw,
                ),
                player1_probability=0.72,
                player2_probability=0.28,
                player1_id=0,
                player2_id=0,
                config=RuntimeConfig(
                    model_path=Path("models/match_winner/lightgbm_elo.joblib"),
                    poll_interval_seconds=20,
                    edge_threshold=0.05,
                    min_model_probability=0.55,
                    min_odds=1.5,
                    max_odds=3.5,
                    default_stake=30.0,
                    bankroll=0.0,
                    kelly_fraction=0.25,
                    dry_run=True,
                    state_path=Path("state.json"),
                    decisions_path=Path("decisions.jsonl"),
                    rl_snapshots_path=Path("rl_snapshots.jsonl"),
                    rl_actions_path=Path("rl_actions.jsonl"),
                    rl_outcomes_path=Path("rl_outcomes.jsonl"),
                    rl_tracker_state_path=Path("rl_tracker_state.json"),
                    rl_market_close_cycles=3,
                ),
            ),
        ]

        executor = FonbetBetExecutor(dry_run=True)
        result = executor.place_express_bet(candidates, stake=30.0)
        bets = result["fonbet_requests"]["bet"]["coupon"]["bets"]
        self.assertEqual(bets[0]["param"], 950)
        self.assertEqual(bets[1]["param"], 1050)

    def test_fonbet_executor_uses_player2_factor_when_side_is_player2(self):
        market = LiveMarket(
            market_id="m2",
            event_id="54321",
            competition="ATP",
            surface="Hard",
            round_name="R32",
            best_of=3,
            tourney_level="tour",
            player1_name="Player One",
            player2_name="Player Two",
            player1_odds=2.55,
            player2_odds=1.46,
            raw={
                "player1_factor_id": 921,
                "player2_factor_id": 923,
                "player1_value": 2.55,
                "player2_value": 1.46,
                "score": "1:0",
                "scopeMarketId": "1600",
            },
        )
        config = RuntimeConfig(
            model_path=Path("models/match_winner/lightgbm_elo.joblib"),
            poll_interval_seconds=20,
            edge_threshold=0.05,
            min_model_probability=0.55,
            min_odds=1.2,
            max_odds=3.5,
            default_stake=100.0,
            bankroll=0.0,
            kelly_fraction=0.25,
            dry_run=True,
            state_path=Path("state.json"),
            decisions_path=Path("decisions.jsonl"),
            rl_snapshots_path=Path("rl_snapshots.jsonl"),
            rl_actions_path=Path("rl_actions.jsonl"),
            rl_outcomes_path=Path("rl_outcomes.jsonl"),
            rl_tracker_state_path=Path("rl_tracker_state.json"),
            rl_market_close_cycles=3,
        )

        candidate = select_candidate(
            market=market,
            player1_probability=0.18,
            player2_probability=0.82,
            player1_id=1,
            player2_id=2,
            config=config,
        )

        executor = FonbetBetExecutor(dry_run=True)
        result = executor.place_bet(candidate)
        bet_payload = result["fonbet_requests"]["bet"]["coupon"]["bets"][0]
        self.assertEqual(candidate.side, "player2")
        self.assertEqual(bet_payload["factor"], 923)
        self.assertEqual(bet_payload["value"], 1.46)

    def test_fonbet_executor_includes_param_for_next_game_market(self):
        market = LiveMarket(
            market_id="63304562:next_game",
            event_id="63304562",
            competition="ITF",
            surface="Hard",
            round_name="R32",
            best_of=3,
            tourney_level="itf",
            player1_name="Saigo R",
            player2_name="Yoshimoto N",
            player1_odds=1.37,
            player2_odds=2.95,
            market_type="next_game_winner",
            raw={
                "player1_factor_id": 1750,
                "player2_factor_id": 1751,
                "player1_value": 1.37,
                "player2_value": 2.95,
                "player1_param": 200,
                "player2_param": 200,
                "score": "0:1",
                "zone": "es",
                "scopeMarketId": "1600",
            },
        )
        config = RuntimeConfig(
            model_path=Path("models/match_winner/lightgbm_elo.joblib"),
            poll_interval_seconds=20,
            edge_threshold=0.05,
            min_model_probability=0.55,
            min_odds=1.2,
            max_odds=3.5,
            default_stake=30.0,
            bankroll=0.0,
            kelly_fraction=0.25,
            dry_run=True,
            state_path=Path("state.json"),
            decisions_path=Path("decisions.jsonl"),
            rl_snapshots_path=Path("rl_snapshots.jsonl"),
            rl_actions_path=Path("rl_actions.jsonl"),
            rl_outcomes_path=Path("rl_outcomes.jsonl"),
            rl_tracker_state_path=Path("rl_tracker_state.json"),
            rl_market_close_cycles=3,
        )
        candidate = select_candidate(
            market=market,
            player1_probability=0.78,
            player2_probability=0.22,
            player1_id=1,
            player2_id=2,
            config=config,
        )
        result = FonbetBetExecutor(dry_run=True).place_bet(candidate)
        bet_payload = result["fonbet_requests"]["bet"]["coupon"]["bets"][0]
        self.assertEqual(bet_payload["factor"], 1750)
        self.assertEqual(bet_payload["param"], 200)
        self.assertEqual(bet_payload["zone"], "es")

    def test_fonbet_executor_refreshes_value_score_and_param_from_betslipinfo(self):
        market = LiveMarket(
            market_id="63282041:1",
            event_id="63282041",
            competition="ATP Challenger",
            surface="Hard",
            round_name="1st set",
            best_of=3,
            tourney_level="challenger",
            player1_name="Mrva M",
            player2_name="Samuel T",
            player1_odds=3.0,
            player2_odds=1.33,
            market_type="next_game_winner",
            raw={
                "player1_factor_id": 1747,
                "player2_factor_id": 1748,
                "player1_value": 3.0,
                "player2_value": 1.33,
                "player1_param": 100,
                "player2_param": 100,
                "score": "0:0",
                "zone": "es",
                "scopeMarketId": "1600",
            },
        )
        config = RuntimeConfig(
            model_path=Path("models/match_winner/lightgbm_elo.joblib"),
            poll_interval_seconds=20,
            edge_threshold=0.05,
            min_model_probability=0.55,
            min_odds=1.2,
            max_odds=4.0,
            default_stake=30.0,
            bankroll=0.0,
            kelly_fraction=0.25,
            dry_run=False,
            state_path=Path("state.json"),
            decisions_path=Path("decisions.jsonl"),
            rl_snapshots_path=Path("rl_snapshots.jsonl"),
            rl_actions_path=Path("rl_actions.jsonl"),
            rl_outcomes_path=Path("rl_outcomes.jsonl"),
            rl_tracker_state_path=Path("rl_tracker_state.json"),
            rl_market_close_cycles=3,
            game_target_offset=3,
        )
        candidate = select_candidate(
            market=market,
            player1_probability=0.7,
            player2_probability=0.3,
            player1_id=1,
            player2_id=2,
            config=config,
        )
        executor = FonbetBetExecutor(dry_run=False)
        executor.api_client = StubFonbetApiClient(
            slip_response={
                "result": "betSlipInfo",
                "bets": [
                    {
                        "event": {"id": 63282041, "score": "1:0"},
                        "factor": {"id": 1747, "v": 4.9, "p": 300},
                    }
                ],
            }
        )
        result = executor.place_bet(candidate)
        bet_payload = executor.api_client.last_bet_payload["coupon"]["bets"][0]
        self.assertEqual(result["status"], "placed")
        self.assertEqual(bet_payload["value"], 4.9)
        self.assertEqual(bet_payload["score"], "1:0")
        self.assertEqual(bet_payload["param"], 300)

    def test_fonbet_executor_retries_unknown_bet_result(self):
        market = LiveMarket(
            market_id="63309103:point:3",
            event_id="63309103",
            competition="ITF",
            surface="Hard",
            round_name="2nd set",
            best_of=3,
            tourney_level="itf",
            player1_name="Sekulic P",
            player2_name="Singh Karan",
            player1_odds=2.0,
            player2_odds=1.72,
            market_type="point_plus_one_winner",
            raw={
                "player1_factor_id": 3010,
                "player2_factor_id": 3011,
                "player1_value": 2.0,
                "player2_value": 1.72,
                "player1_param": 720899,
                "player2_param": 720899,
                "score": "5:5",
                "zone": "es",
                "scopeMarketId": "1600",
            },
        )
        config = RuntimeConfig(
            model_path=Path("models/match_winner/lightgbm_elo.joblib"),
            poll_interval_seconds=20,
            edge_threshold=0.01,
            min_model_probability=0.5,
            min_odds=1.2,
            max_odds=3.5,
            default_stake=30.0,
            bankroll=0.0,
            kelly_fraction=0.25,
            dry_run=False,
            state_path=Path("state.json"),
            decisions_path=Path("decisions.jsonl"),
            rl_snapshots_path=Path("rl_snapshots.jsonl"),
            rl_actions_path=Path("rl_actions.jsonl"),
            rl_outcomes_path=Path("rl_outcomes.jsonl"),
            rl_tracker_state_path=Path("rl_tracker_state.json"),
            rl_market_close_cycles=3,
        )
        candidate = select_candidate(
            market=market,
            player1_probability=0.39,
            player2_probability=0.61,
            player1_id=1,
            player2_id=2,
            config=config,
        )
        executor = FonbetBetExecutor(dry_run=False, bet_result_retries=2, bet_result_retry_delay_seconds=0.0)
        executor.api_client = StubFonbetApiClient(
            slip_response={"result": "betSlipInfo", "bets": []},
            bet_result_response=[
                {"result": "error", "errorCode": 200, "errorMessage": "Request result currently unknown"},
                {"result": "couponResult", "coupon": {"resultCode": 0, "regId": 1}},
            ],
        )
        result = executor.place_prepared_bet(candidate, {}, {"result": "betSlipInfo", "bets": []})
        self.assertEqual(result["bet_result_response"]["coupon"]["resultCode"], 0)
        self.assertEqual(len(result["bet_result_attempts"]), 2)

    def test_markov_game_model_prefers_server(self):
        model = MarkovGameModel()
        probability = model.predict_next_game(
            {
                "p1_hold_rate": 0.78,
                "p2_hold_rate": 0.63,
                "p1_break_rate": 0.29,
                "p2_break_rate": 0.18,
            },
            {"serveT": 1},
        )
        self.assertGreater(probability, 0.6)
        self.assertEqual(infer_server_side({"serveT": "2"}), "player2")

    def test_markov_point_model_prefers_server(self):
        model = MarkovGameModel()
        probability = model.predict_point_plus_one(
            {
                "p1_hold_rate": 0.8,
                "p2_hold_rate": 0.62,
                "p1_break_rate": 0.31,
                "p2_break_rate": 0.2,
            },
            {"serveT": 1},
        )
        self.assertGreater(probability, 0.55)

    def test_bankroll_bandit_policy_reduces_stake_after_negative_rewards(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            outcomes = Path(tmp_dir) / "rl_outcomes.jsonl"
            outcomes.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "market_type": "next_game_winner",
                            "odds_taken": 1.8,
                            "reward": -0.1,
                        }
                    )
                    for _ in range(12)
                )
                + "\n",
                encoding="utf-8",
            )
            policy = BankrollBanditPolicy(outcomes_path=outcomes, bankroll=1000.0, min_samples=10)
            market = LiveMarket(
                market_id="m3",
                event_id="e3",
                competition="ITF",
                surface="Hard",
                round_name="R32",
                best_of=3,
                tourney_level="itf",
                player1_name="A",
                player2_name="B",
                player1_odds=1.8,
                player2_odds=2.0,
                market_type="next_game_winner",
            )
            candidate = select_candidate(
                market=market,
                player1_probability=0.7,
                player2_probability=0.3,
                player1_id=1,
                player2_id=2,
                config=RuntimeConfig(
                    model_path=Path("models/match_winner/lightgbm_elo.joblib"),
                    poll_interval_seconds=20,
                    edge_threshold=0.05,
                    min_model_probability=0.55,
                    min_odds=1.2,
                    max_odds=3.5,
                    default_stake=100.0,
                    bankroll=1000.0,
                    kelly_fraction=0.25,
                    dry_run=True,
                    state_path=Path("state.json"),
                    decisions_path=Path("decisions.jsonl"),
                    rl_snapshots_path=Path("rl_snapshots.jsonl"),
                    rl_actions_path=Path("rl_actions.jsonl"),
                    rl_outcomes_path=Path("rl_outcomes.jsonl"),
                    rl_tracker_state_path=Path("rl_tracker_state.json"),
                    rl_market_close_cycles=3,
                ),
            )
            self.assertIsNotNone(candidate)
            adjusted = policy.recommend_stake(candidate, market.market_type)
            self.assertLess(adjusted, candidate.stake)

    def test_bankroll_bandit_policy_selects_discrete_rl_stake(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            outcomes = Path(tmp_dir) / "rl_outcomes.jsonl"
            outcomes.write_text("", encoding="utf-8")
            policy = BankrollBanditPolicy(outcomes_path=outcomes, bankroll=150.0, min_samples=1)
            market = LiveMarket(
                market_id="m4",
                event_id="e4",
                competition="ITF",
                surface="Hard",
                round_name="R32",
                best_of=3,
                tourney_level="itf",
                player1_name="A",
                player2_name="B",
                player1_odds=1.85,
                player2_odds=2.1,
                market_type="match_winner",
            )
            config = RuntimeConfig(
                model_path=Path("models/match_winner/lightgbm_elo.joblib"),
                poll_interval_seconds=20,
                edge_threshold=0.05,
                min_model_probability=0.55,
                min_odds=1.2,
                max_odds=3.5,
                default_stake=30.0,
                bankroll=150.0,
                kelly_fraction=0.25,
                dry_run=True,
                state_path=Path("state.json"),
                decisions_path=Path("decisions.jsonl"),
                rl_snapshots_path=Path("rl_snapshots.jsonl"),
                rl_actions_path=Path("rl_actions.jsonl"),
                rl_outcomes_path=Path("rl_outcomes.jsonl"),
                rl_tracker_state_path=Path("rl_tracker_state.json"),
                rl_market_close_cycles=3,
            )
            candidate = select_candidate(
                market=market,
                player1_probability=0.66,
                player2_probability=0.34,
                player1_id=1,
                player2_id=2,
                config=config,
            )
            self.assertIsNotNone(candidate)
            recommended = policy.recommend([candidate], market.market_type)
            self.assertIsNotNone(recommended)
            self.assertIn(recommended.stake, {30.0, 60.0, 90.0})

    def test_settle_outcome_record_updates_bankroll_for_match_winner(self):
        record = {
            "market_type": "match_winner",
            "side": "player1",
            "stake": 60.0,
            "odds_taken": 1.8,
        }
        final_snapshot = {"live_score": "2:0", "best_of": 3}
        settled = settle_outcome_record(record, final_snapshot=final_snapshot, bankroll_before=1000.0)
        self.assertEqual(settled["status"], "won")
        self.assertEqual(settled["profit"], 48.0)
        self.assertEqual(settled["bankroll_after"], 1048.0)
        self.assertAlmostEqual(settled["reward"], 0.048, places=4)

    def test_settle_outcome_record_updates_bankroll_for_next_game_winner(self):
        record = {
            "market_type": "next_game_winner",
            "target_game_number": 11,
            "side": "player2",
            "stake": 30.0,
            "odds_taken": 2.1,
            "opening_snapshot": {"live_score": "5:5"},
        }
        final_snapshot = {"live_score": "5:6", "best_of": 3}
        settled = settle_outcome_record(record, final_snapshot=final_snapshot, bankroll_before=1000.0)
        self.assertEqual(settled["status"], "won")
        self.assertEqual(settled["winner_side"], "player2")
        self.assertEqual(settled["profit"], 33.0)
        self.assertEqual(settled["bankroll_after"], 1033.0)
        self.assertAlmostEqual(settled["reward"], 0.033, places=4)

    def test_settle_outcome_record_updates_bankroll_for_set_total_over_under(self):
        record = {
            "market_type": "set_total_over_under",
            "side": "player1",
            "stake": 30.0,
            "odds_taken": 1.9,
            "opening_snapshot": {
                "live_score": "4:4",
                "player1_param": 950,
                "state_features": {"total_line": 9.5},
            },
        }
        final_snapshot = {"live_score": "6:4", "best_of": 3}
        settled = settle_outcome_record(record, final_snapshot=final_snapshot, bankroll_before=1000.0)
        self.assertEqual(settled["status"], "won")
        self.assertEqual(settled["winner_side"], "player1")
        self.assertEqual(settled["profit"], 27.0)
        self.assertEqual(settled["bankroll_after"], 1027.0)
        self.assertAlmostEqual(settled["reward"], 0.027, places=4)

    def test_settle_outcome_record_updates_bankroll_for_point_plus_one_winner(self):
        record = {
            "market_type": "point_plus_one_winner",
            "side": "player2",
            "stake": 30.0,
            "odds_taken": 1.8,
            "opening_snapshot": {
                "live_score": "0:0",
                "live_comment": "(15-00*)",
                "target_point_number": 3,
                "state_features": {"target_point_number": 3},
            },
        }
        final_snapshot = {"live_score": "0:0", "live_comment": "(15-30*)", "best_of": 3}
        settled = settle_outcome_record(record, final_snapshot=final_snapshot, bankroll_before=1000.0)
        self.assertEqual(settled["status"], "won")
        self.assertEqual(settled["winner_side"], "player2")
        self.assertEqual(settled["profit"], 24.0)
        self.assertEqual(settled["bankroll_after"], 1024.0)
        self.assertAlmostEqual(settled["reward"], 0.024, places=4)

    def test_bankroll_policy_has_no_player1_bias(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            outcomes_path = Path(tmp_dir) / "outcomes.jsonl"
            policy = BankrollBanditPolicy(outcomes_path=outcomes_path, bankroll=0.0)
            market = LiveMarket(
                market_id="g1",
                event_id="e1",
                competition="ITF",
                surface="Hard",
                round_name="1st set",
                best_of=3,
                tourney_level="itf",
                player1_name="A",
                player2_name="B",
                player1_odds=2.0,
                player2_odds=2.0,
                market_type="next_game_winner",
                raw={},
            )
            player2_candidate = ScoredSelection(
                market=market,
                side="player2",
                player_name="B",
                model_probability=0.7,
                implied_probability=0.5,
                edge=0.2,
                odds=2.0,
                stake=30.0,
                player_id=2,
            )
            player1_candidate = ScoredSelection(
                market=market,
                side="player1",
                player_name="A",
                model_probability=0.7,
                implied_probability=0.5,
                edge=0.2,
                odds=2.0,
                stake=30.0,
                player_id=1,
            )
            recommended = policy.recommend([player2_candidate, player1_candidate], market.market_type)
            self.assertIsNotNone(recommended)
            self.assertEqual(recommended.side, "player2")


if __name__ == "__main__":
    unittest.main()
