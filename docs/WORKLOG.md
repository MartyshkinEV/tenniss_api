# Worklog

## 2026-03-15

### Repository Snapshot
- Uncommitted refactor in progress across config, training scripts, prediction scripts, and tests.
- New untracked files include `scripts/_bootstrap.py`, `scripts/compare_models.py`, `scripts/run_live_betting.py`, `scripts/train_logreg_elo_model.py`, `src/live/`, `tests/test_live_runtime.py`, and `tests/test_predict_match.py`.

### What Changed Recently
- `config.py` now owns environment loading, project paths, live runtime settings, and model artifact resolution.
- README was updated to describe the new commands and the current architecture.
- Prediction/training scripts were aligned around shared model resolution and artifact directories.
- A new live betting flow was added:
  - feed clients for file/Fonbet/Spoyer
  - runtime selection logic
  - Fonbet bet executor with dry-run support
- Tests were added/updated to cover config, feature building, prediction flow, and live runtime behavior.

### What Still Needs Verification
- End-to-end test pass status in the current environment.
- Whether live runtime works correctly with real provider payloads.
- Whether model/report scripts work with the locally available artifacts.

### Live Betting Context Captured From Conversation
- New Fonbet-related env/config fields were introduced for:
  - `FONBET_COUPON_INFO_URL`
  - `FONBET_BET_URL`
  - `FONBET_BET_RESULT_URL`
  - `FONBET_ORIGIN`
  - `FONBET_REFERER`
  - `FONBET_USER_AGENT`
  - `FONBET_COOKIE`
- `src/live/fonbet.py` was extended with:
  - `_deep_merge(...)`
  - `FonbetApiClient`
  - multi-request payload building for `betSlipInfo`, `bet`, `betResult`
  - dry-run output of all Fonbet request payloads
- Tests already passed at that stage:
  - `venv/bin/python -m unittest tests.test_live_runtime tests.test_config`
- User-provided captured payload fragments confirmed:
  - `betSlipInfo`/`betResult` metadata includes `lang`, `fsid`, `sysId`, `clientId`, `CDI`, `deviceId`
  - `bet` request includes `requestId`, `lang`, `fsid`, `sysId`, `clientId`, `coupon`
  - `coupon` includes `amount`, `flexBet`, `flexParam`, `mirror`, `bets`
  - live selection uses `zone: "lv"`
  - prematch uses `zone: "sp"`
  - live + prematch in the same coupon is rejected by Fonbet
- Stopping point from the conversation:
  - next implementation step was to harden the exact payload builder around the confirmed Fonbet fields
  - remaining missing artifacts were full request/response bodies for `betSlipInfo`, `betResult`, and a successful single live tennis bet

### 2026-03-15 Follow-Up
- Hardened `src/live/fonbet.py` for a stricter single-live coupon shape:
  - session defaults now include `lang`, `fsid`, `sysId`, `clientId`, `CDI`, `deviceId`
  - coupon defaults now include `amount`, `flexBet`, `flexParam`, `mirror`, `bets`
  - selection payload now targets single live bets with `event`, `factor`, `value`, optional `score`, and `zone`
- Added matching env/config support in `config.py` and `.env.example`:
  - `FONBET_LANG`
  - `FONBET_FSID`
  - `FONBET_SYS_ID`
  - `FONBET_CLIENT_ID`
  - `FONBET_CDI`
  - `FONBET_DEVICE_ID`
  - `FONBET_MIRROR`
  - `FONBET_FLEX_BET`
  - `FONBET_FLEX_PARAM`
- Validation status:
  - `venv/bin/python -m unittest tests.test_live_runtime tests.test_config` passed
- Runtime blocker for a real first bet:
  - current `.env` does not yet contain the required live Fonbet session values or endpoint URLs

### 2026-03-15 Bet Chain Update
- Updated the live Fonbet chain to match the newly provided HAR details:
  - added `FONBET_BET_REQUEST_ID_URL`
  - added explicit `betRequestId` call between `betSlipInfo` and `bet`
  - switched `betSlipInfo` payload shape to:
    - `lang`
    - `fsid`
    - `sysId`
    - `clientId`
    - `CDI`
    - `deviceId`
    - `bets: [{eventId, factorId, old}]`
    - optional `scopeMarketId`
- `bet` payload remains coupon-based with a single live tennis selection.
- Tests still pass after this update:
  - `venv/bin/python -m unittest tests.test_live_runtime tests.test_config`
- Current real-world blocker is not code structure:
  - the runtime still needs actual `.env` values and a live tennis market carrying `factorId` and preferably `scopeMarketId`

### 2026-03-15 First Real Bet Confirmed
- Verified external Fonbet session with `POST /session/info`:
  - `clientId=20211191`
  - `scopeMarket=1600`
  - `liveBlocked=false`
  - `betBlocked=false`
  - balance before bet: `8570 RUB`
- Pulled live market data from:
  - `GET /ma/line/liveEvents`
  - `GET /ma/events/list?lang=en&version=72508454342&scopeMarket=1600`
- Identified confirmed live tennis market:
  - event `63203793`
  - `Gorbachyov Aleksandr` vs `Parypchik Dmitrii`
  - factor `921`
  - precheck `betSlipInfo` returned live score `1:0` and price `2.55`
  - minimum stake `30 RUB`
- Successful real bet flow:
  - `POST /coupon/betSlipInfo` -> accepted
  - `POST /coupon/betRequestId` -> `requestId=12332862698`
  - `POST /coupon/bet` -> `betDelay=2000`
  - `POST /coupon/betResult` -> `resultCode=0`
- Confirmed registration details for the first real bet:
  - `regId=5347288384`
  - `checkCode=ND-M5-TP`
  - stake `30 RUB`
  - odds `2.55`
  - balance after bet: `8540 RUB`

### 2026-03-15 Runtime Integration Follow-Up
- Added real Fonbet catalog parsing in `src/live/fonbet.py`:
  - parses `events`, `eventMiscs`, `customFactors`, `sports`
  - extracts live tennis root events from Fonbet segment hierarchy
  - maps player1 to `factorId=921` and player2 to `factorId=923`
  - stores side-specific factor/value metadata in `LiveMarket.raw`
- Updated payload building to be side-aware:
  - if runtime selects `player2`, bet builder now uses the correct opposing factor/value instead of always `921`
- Added config support for:
  - `FONBET_SCOPE_MARKET_ID`
- Validation:
  - `venv/bin/python -m unittest tests.test_live_runtime tests.test_config` passed
  - `scripts/run_live_betting.py --once --feed-file /tmp/fonbet_events_list.json` successfully parsed the current Fonbet catalog
- Current automation blocker:
  - runtime still skips all currently visible live tennis markets because those players are not present in the local historical ATP dataset used for feature generation/model scoring
  - examples skipped at runtime:
    - `Saigo R`
    - `Morvayova V`
    - `Okamura K`
    - `Gorbachyov Aleksandr`

### 2026-03-15 Dataset Expansion
- Generalized CSV discovery away from ATP-only:
  - match discovery now accepts compatible `atp_matches*` and `wta_matches*` tables
  - player discovery now accepts multiple `*_players.csv`
  - ranking discovery now accepts multiple `*_rankings*.csv`
  - point-by-point `pbp_matches_*` files are explicitly excluded
- Added namespace-safe `player_id` offsets:
  - ATP keeps original ids
  - WTA ids are offset to avoid cross-circuit collisions
- Updated both local fallback pipeline and DB loader:
  - `src/data/pipeline.py`
  - `scripts/load_atp.py`
- Downloaded additional dataset:
  - `tennis_wta/` cloned from Jeff Sackmann `tennis_wta`
- Post-expansion discovery counts:
  - compatible match CSVs: `252`
  - player CSVs: `atp_players.csv`, `wta_players.csv`
  - ranking CSVs: `13`
- Rebuilt local fallback `player_match_stats.joblib` on ATP+WTA:
  - shape: `(3337250, 34)`
  - unique players: `58609`
  - date range: `1967-12-25` to `2024-12-18`
- Result on live runtime after the expanded `player_match_stats` rebuild:
  - WTA/ITF live markets no longer fail immediately on name lookup
  - the remaining skip in the sampled feed was `Gorbachyov Aleksandr` from Liga Pro, which is outside ATP/WTA coverage
- Environment status:
  - PostgreSQL connection was unavailable, so expansion was applied through local cache fallback rather than DB tables

### 2026-03-15 RL Dataset Logging
- Added RL-specific append-only logging paths in config:
  - `LIVE_RL_SNAPSHOTS_PATH`
  - `LIVE_RL_ACTIONS_PATH`
- Added `RLDatasetLogger` in `src/live/runtime.py`
- Runtime now records on every polling cycle:
  - snapshot records for every seen market
  - action records for every decision branch
- Snapshot payload currently includes:
  - timestamp
  - event/market metadata
  - live score/comment/delay/server
  - side odds and factor ids
  - model probabilities
  - selected action context
  - full feature-state used for decision making
- Action payload currently includes:
  - timestamp
  - event/market ids
  - action label (`skip`, `no_bet`, `duplicate`, `bet`, `bet_dry_run`)
  - optional candidate metadata
  - optional raw bet result payload
- Validation:
  - tests passed after the change
  - dry-run `scripts/run_live_betting.py --once --feed-file /tmp/fonbet_events_list.json` created:
    - `artifacts/live_betting/rl_snapshots.jsonl`
    - `artifacts/live_betting/rl_actions.jsonl`
- Current RL logging status:
  - suitable as the first stage for offline RL / imitation / contextual bandit training
  - reward/outcome settlement layer is still not implemented yet

### 2026-03-15 RL Outcome Tracker
- Added new config paths:
  - `LIVE_RL_OUTCOMES_PATH`
  - `LIVE_RL_TRACKER_STATE_PATH`
  - `LIVE_RL_MARKET_CLOSE_CYCLES`
- Added persistent `RLOutcomeTracker` in `src/live/runtime.py`
- Runtime now tracks:
  - last seen cycle per live event
  - last snapshot per event
  - pending bet outcomes by event
- Behavior:
  - on `bet` / `bet_dry_run` the runtime writes a pending outcome record
  - if a pending market disappears from the live feed for more than the configured grace window, it is marked as `market_closed_unsettled`
- Validation:
  - tests pass after the tracker changes
  - dry-run created:
    - `artifacts/live_betting/rl_outcomes.jsonl`
    - `artifacts/live_betting/rl_tracker_state.json`
  - tracker state now persists latest snapshots and seen-cycle metadata across runs
- Current limitation:
  - this is lifecycle tracking, not true settlement yet
  - real numeric `reward` is still `null` until a settlement source is connected

### 2026-03-15 Live Tennis Game Bet Procedure Confirmed
- Confirmed the reliable workflow for live tennis game bets:
  1. fetch `GET /ma/events/event?lang=en&version=...&eventId=...&scopeMarket=1600`
  2. inspect `customFactors` in the returned payload
  3. identify target game markets by `p` and `pt`
  4. only after that, verify the chosen `factorId` through `POST /coupon/betSlipInfo`
  5. build and send `betRequestId -> bet -> betResult`
- Confirmed mapping for tennis game number:
  - game 1 -> `p=100`, `pt="1"`
  - game 3 -> `p=300`, `pt="3"`
  - game 4 -> `p=400`, `pt="4"`
  - game 5 -> `p=500`, `pt="5"`
  - game 6 -> `p=600`, `pt="6"`
  - game 7 -> `p=700`, `pt="7"`
- Important rule:
  - do not start from `betSlipInfo` alone when discovering game markets
  - first locate the factor candidates in `events/event` by `p/pt`
  - then use `betSlipInfo` only to confirm which factor is currently live, what player caption it maps to, and the refreshed price/score
- Confirmed examples from real traffic and live checks:
  - `63319046`, `p=500`:
    - `1747` -> `Game 5 - Smith Alana`
    - `1748` -> `Game 5 - Bartashevich Y`
  - `63319501`, `p=400`:
    - `9961` -> `Game 4 - Juan Mano A`
    - `9962` -> `Game 4 - Arnaboldi F`
  - `63319501`, `p=600`:
    - `1750` -> `Game 6 - Juan Mano A`
    - `1751` -> `Game 6 - Arnaboldi F`
  - `63299287`, `p=100`:
    - `1747` -> `Game 1 - Borrelli L`
    - `1748` -> `Game 1 - Wessels L`
- Two successful real game-bet registrations were confirmed:
  - `requestId=12336194507` -> accepted as `regId=5374979252`, `checkCode=NK-T7-E3`
    - selection: `Game 5 - Smith Alana`
    - event: `63319046`
    - odds: `1.19`
    - amount: `30`
    - final result saved to `artifacts/live_betting/bet_result_12336194507.json`
  - `requestId=12336334160` -> accepted as `regId=5376143455`, `checkCode=EW-MU-11`
    - selection: `Game 1 - Borrelli L`

## 2026-03-16 Live Betting Orchestrator Session

### User Goal
- Keep the live betting bot running continuously.
- Continuously discover new live events and add new bets aggressively.
- Allow multiple bets on the same match, but do not repeat the exact same market/game/side.
- Log the full decision process.
- Add periodic retraining from accumulated live data.

### Logs And Runtime Findings
- The primary runtime journal is `artifacts/live_betting/auto_live_event_betting.jsonl`.
- The plain `.log` files were mostly empty; the useful execution trace is in `jsonl`.
- Multiple times the orchestrator was found stopped while the PID/state files still existed.
- A stale duplicate-protection bug existed:
  - `placed_selection_keys` survived longer than real active exposure
  - this caused endless `duplicate_skipped` decisions even when the old bet should no longer block new ones
- Separate state pressure was also observed:
  - active bets could remain in `auto_live_event_betting_state.json` long enough to reduce fresh bet flow
  - even when exposure was low, exact duplicate keys still blocked identical `match/game/side` selections

### Code Changes Made
- Fixed duplicate-protection cleanup in `scripts/run_auto_live_event_betting.py` so keys are no longer kept forever.
- Switched the batch orchestrator from partial event rotation to processing all live root tennis events each cycle by default.
- Added richer decision journaling:
  - `cycle_started`
  - `event_scored`
  - `cycle_ranked`
  - `placement_attempt`
  - `placement_deferred`
  - final `event_cycle`
  - `heartbeat`
- Increased aggressiveness:
  - `max_bets_per_cycle` default changed from `3` to `5`
  - `min_acceptance_probability` default changed from `0.55` to `0.50`
  - `active_bet_ttl_minutes` default changed from `20` to `6`
- Split exposure tracking from exact duplicate protection:
  - `active_bets` now serve short-lived exposure accounting
  - `recent_selection_keys` now serve exact duplicate blocking for `market/game/side`
- Added new exact-duplicate control:
  - `--duplicate-key-ttl-minutes` default `180`
  - this allows more bets on the same match while still blocking exact repeat bets
- Added periodic retraining hooks:
  - `--retrain-every-minutes` default `30`
  - `--retrain-min-rows` default `200`
  - retraining runs:
    - `scripts/train_leg_acceptance_model.py`
    - `scripts/train_rl_policy.py`
  - retrain status is written to the journal as:
    - `retrain_started`
    - `retrain_finished`
    - `retrain_failed`

### Aggressive Discovery Upgrade
- Refactored event scoring from “one best candidate per event” to a per-event candidate pool.
- Enabled aggressive market coverage in the batch orchestrator:
  - `next_game_winner`
  - `set_total_over_under`
  - `point_plus_one_winner`
- Added target-fill behavior:
  - `--target-active-bets` default `10`
  - placement budget is now driven by both:
    - `max_bets_per_cycle`
    - deficit to target active bets
- The ranking pool is now global across all currently scored candidates, not one-candidate-per-event only.

### Operational Issues Observed During The Session
- The orchestrator sometimes stopped unexpectedly after restart attempts; several restarts were performed during debugging.
- During healthy cycles, the limiting factor was often not exposure but low candidate supply:
  - `live_events_total` could be around `12-15`
  - `candidate_events_total` could still be only `2-4`
- Before the aggressive multi-market refactor, the bot frequently found candidates but skipped them as exact duplicates:
  - examples included repeated keys such as `63324997:5:player1:1747:500`
  - and `63341527:7:player1:1753:700`

### Current Task Status
- Implemented:
  - all-live event polling
  - stronger decision journaling
  - exact duplicate blocking by `market/game/side`
  - aggressive placement defaults
  - periodic retraining hooks
  - multi-market candidate pool logic in the orchestrator code
- Still needs runtime verification:
  - a full clean cycle from the newest aggressive multi-market version should be observed in `auto_live_event_betting.jsonl`
  - confirm that `set_total_over_under` and `point_plus_one_winner` actually contribute real placed bets in production, not just `next_game_winner`
  - confirm the long-running process remains alive after restart and does not silently stop

### Files Touched In This Session
- `scripts/run_auto_live_event_betting.py`
- `docs/WORKLOG.md`

### Main Remaining Risk
- Even after aggressive tuning, sustained bet volume still depends on real live-market supply and model filters.
- If live feeds provide too few valid opportunities, the next tuning step should be:
  - lower thresholds further, or
  - add additional market families to the aggressive candidate pool, or
  - add a background supervisor/systemd unit dedicated to `run_auto_live_event_betting.py --send-bet`.
    - event: `63299287`
    - odds: `1.85`
    - amount: `30`
    - final result saved to `artifacts/live_betting/bet_result_12336334160.json`

### 2026-03-15 Conversation Continuation
- User clarified the live tennis discovery rule for game markets:
  - earlier rule was superseded by a stricter user clarification
  - current valid rule is to jump one game ahead, not to check both adjacent futures
  - examples confirmed by the user:
    - if the current game is `1`, target game is `3`
    - if the current game is `3`, target game is `5`
    - if the current game is `4`, target game is `6`
  - shorthand: bet on the game after next (`current + 2`)
- Confirmed market-number mapping in `ma/events/event`:
  - target game `4` -> `p=400`, `pt="4"`
  - target game `5` -> `p=500`, `pt="5"`
- Live example discussed in the conversation:
  - subevent `63323110`
  - currently live market only for target game `3`
  - score `2:0`
  - comment `(40*-00)`
  - confirmed through `betSlipInfo`:
    - `1753` -> `Evans K @ 1.02`, `param=300`
    - `1754` -> `Miroshnichenko V @ 10.0`, `param=300`
- Important implementation note from the user:
  - when querying `ma/events/event`, `eventId` must be replaced for each match with that match's root event id
  - using the sample link with a fixed `eventId=63304637` is only valid as an example and must not remain constant in code or operator workflow
- Current coding direction at the stopping point:
  - inspect the live parser/runtime to ensure it does not hardcode a sample `eventId`
  - adjust game-market filtering so the runtime can consider both `current+1` and `current+2`, not only a single exact target
- Additional note from the user:
  - PostgreSQL already contains a table with online/live events and it should be treated as an available source of truth during manual discovery
  - in the current codebase, the known table used for current live tennis root ids is `fonbet_tennis_live_events_latest`
- Manual check status:
  - selected root event id from Postgres: `63292146` (`Fenty A` vs `Rottgering M`)
  - live details request used:
    - `https://line-lb61-w.bk6bba-resources.com/ma/events/event?lang=en&version=0&eventId=63292146&scopeMarket=1600`
  - response at the time of the check contained only packet metadata and no `events`, `eventMiscs`, or `customFactors`
  - result: no current next-game market could be discovered for this event through `ma/events/event`
- User correction recorded:
  - the previous manual check used the wrong request template
  - user-provided correct example template:
    - `https://line-lb54-w.bk6bba-resources.com/ma/events/event?lang=en&version=72543129024&eventId=63301294&scopeMarket=1600`
  - user indicated that for the manual lookup we should insert the desired event id into that template, specifically replacing `eventId=63301294` with the live event being checked, such as `63292146`
  - this correction means the host/version pair from the working captured link matters for the user's current manual verification workflow
  - additional clarification from the user:
    - when searching a new event, only `eventId` should be changed in that URL
    - the rest of the link, including host `line-lb54-w`, `version=72543129024`, `lang=en`, and `scopeMarket=1600`, should stay unchanged for this manual workflow
- Later live example provided by the user:
  - root event id `63300309`
  - exact working link shared by the user:
    - `https://line-lb54-w.bk6bba-resources.com/ma/events/event?lang=en&version=72534781357&eventId=63300309&scopeMarket=1600`
  - this should be treated as the current concrete reference URL for manual inspection of game markets on that match
- URL templating rule clarified by the user:
  - to avoid confusion, the manual lookup URL should be written with a variable placeholder instead of a hardcoded number
  - canonical template for the current workflow:
    - `https://line-lb54-w.bk6bba-resources.com/ma/events/event?lang=en&version=72534781357&eventId={event_id}&scopeMarket=1600`
  - when assigning a new live event, only `{event_id}` changes, and the actual value in the URL must change together with that assignment
    - event: `63299287`
    - odds: `2.95`
    - amount: `30`
    - final result saved to `artifacts/live_betting/bet_result_12336334160.json`
- Practical note for implementation:
  - the current generic game-market extractor in `src/live/fonbet.py` only covers part of the available Fonbet game factor families
  - for real execution, raw `events/event` discovery by `p/pt` is currently the source of truth

### 2026-03-15 Async Per-Event ML Watcher
- Added `scripts/run_event_ml_watch.py`
- Purpose:
  - run asynchronous analysis loops per live tennis root event
  - fetch `ma/events/event` independently for each live match
  - score available markets continuously with the existing ML runtime
  - write top candidates per event into `artifacts/live_betting/event_ml_watch.jsonl`
- Current behavior:
  - default event source is `fonbet_tennis_live_events_latest WHERE level = 1`
  - one async task is created per live event
  - each task polls its own event payload on a configurable interval
  - task output includes:
    - root event id
    - market id / market event id
    - market type
    - live score/comment
    - selected side / factor / param
    - model probability / implied probability / edge / ranking score
- Added reusable `FonbetEventDetailsClient` in `src/live/fonbet.py`
- Deliberate limitation in the first version:
  - `--autobet` is reserved but automatic placement is still intentionally not wired in the watcher
  - this version is for continuous asynchronous discovery/scoring first, not autonomous execution yet

### 2026-03-15 Next-Game Markov + RL Policy
- Finished the missing runtime wiring for RL outcome tracking:
  - `run_cycle()` now starts a tracker cycle, records seen events, writes pending outcomes after bets, and closes missing markets into `market_closed_unsettled`
- Added game-market support for Fonbet:
  - `LiveMarket` now carries `market_type`
  - Fonbet catalog parser now emits:
    - `match_winner` markets via factors `921/923`
    - `next_game_winner` markets via factors `1750/1751` when present
  - next-game payload builder now includes `param` and preserves `zone="es"`
- Added Markov-based next-game model:
  - `src/live/markov.py`
  - estimates next-game win probability from historical hold/break rates plus serving side
- Added bankroll-aware RL policy layer:
  - `src/live/policy.py`
  - reads `rl_outcomes.jsonl`
  - downweights or slightly upweights stake by `market_type + odds bucket`
- Added training/export entrypoints:
  - `scripts/train_game_markov_model.py`
  - `scripts/train_rl_policy.py`
  - wrappers in `src/training/train.py`

### 2026-03-15 Manual Live Express Procedure
- Confirmed working manual path for real multi-leg live tennis expresses when the automatic runner is too rigid or when odds must be refreshed immediately before placement.
- Successful real 6-leg express confirmed:
  - `regId=5361347039`
  - `checkCode=24-CH-1E`
  - accepted after refreshing each leg through `betSlipInfo`
- Working procedure:
  - fetch a single frozen Fonbet live snapshot from `ma/events/list?...scopeMarket=1600`
  - score all visible live markets through the existing runtime models (`match_winner`, `point_plus_one_winner`, `set_total_over_under`)
  - keep the best candidate per `event_id`
  - shortlist more events than needed (`8-10` for a target coupon size of `6`) so blocked legs can be replaced without re-analysing a new feed
  - for each candidate combination:
    - call `betSlipInfo` first
    - rebuild the outgoing coupon from the returned live `factor.id`, `factor.v`, `factor.p`, and current event `score`
    - request a fresh `betRequestId`
    - send `bet`
    - poll `betResult` until `couponResult`
  - if `resultCode != 0`, move to the next combination from the same frozen shortlist instead of re-fetching the feed
- Practical notes from the successful run:
  - this approach preserves the analysis window better than a long-running auto loop because pricing is refreshed only at the final pre-bet step
  - `betSlipInfo` can return updated odds/params that differ materially from the model-snapshot odds; the outgoing coupon should use the refreshed values
  - a mixed coupon across totals / point markets / match winner was accepted once all six legs were rebuilt from `betSlipInfo`

### 2026-03-15 Acceptance Model Bootstrap
- Added `src/training/live_acceptance_dataset.py`:
  - builds a leg-level acceptance dataset from `artifacts/live_betting/rl_actions.jsonl`
  - extracts requested odds, accepted odds, slip-factor metadata, coupon size, delay, result code
- Added `scripts/train_leg_acceptance_model.py`
  - writes `artifacts/models/leg_acceptance_model.joblib`
  - writes `artifacts/models/leg_acceptance_model.json`
- Added `tests/test_live_acceptance_training.py`
- Initial intent:
  - rank live candidates by both ML edge and probability of clean Fonbet acceptance
  - support later coupon-level survival modelling
- Validation:
  - `venv/bin/python -m unittest tests.test_live_runtime tests.test_config` -> `OK`
  - `venv/bin/python scripts/train_game_markov_model.py` created `artifacts/models/game_markov_model.joblib`
  - `venv/bin/python scripts/train_rl_policy.py` created `artifacts/models/rl_bankroll_policy.json`
- Current blocker for a real next-game bet:
  - `.env` in the current workspace does not contain any live Fonbet endpoints/session values
  - specifically empty: `FONBET_SESSION_INFO_URL`, `FONBET_FEED_URL`, `FONBET_COUPON_INFO_URL`, `FONBET_BET_REQUEST_ID_URL`, `FONBET_BET_URL`, `FONBET_BET_RESULT_URL`, `FONBET_COOKIE`, `FONBET_FSID`, `FONBET_CLIENT_ID`, `FONBET_DEVICE_ID`
- Practical next step:
  - populate `.env` with the live Fonbet session values and then run a real `next_game_winner` dry-run/live attempt against the current feed

### Resume Hint
- Start from `docs/SESSION.md`, then check `git diff --stat` and run targeted tests before making new edits.

### 2026-03-15 Point Trajectory Logging
- Added `LIVE_POINT_TRAJECTORIES_PATH` to config and `.env.example`
- Extended `RLDatasetLogger` with `point_trajectories.jsonl`
- Runtime now writes point-only trajectory records for:
  - initial scoring / no-edge observations
  - duplicate suppression
  - refresh failures
  - refresh invalidation
  - refreshed candidates
  - bet / dry-run bet records
  - closed point-market lifecycle records
- Added tests for:
  - point JSONL append behavior
  - runtime point trajectory logging on a dry-run point market
- Validation:
  - `venv/bin/python -m unittest tests.test_live_runtime tests.test_config` -> `OK`
# 2026-03-15 - Historical point-by-point source integrated

- Added `discover_pbp_csv_files()` usage to the local data pipeline and introduced `load_historical_points()` cache builder backed by `tennis_pointbypoint`.
- Added `src/data/point_parser.py` integration into `src/data/pipeline.py` so Sackmann `pbp` strings expand into point-level rows with score/server metadata.
- Added `scripts/build_historical_points.py` for cache rebuild and `tests/test_point_parser.py` to cover regular games, server alternation, tiebreak parsing, and point-frame construction.
- Verified test suite: `venv/bin/python -m unittest tests.test_point_parser tests.test_live_point_training tests.test_live_runtime tests.test_config` -> `OK`.
- Full `historical_points` cache build is CPU/memory-heavy and should be treated as a batch step; code path is in place and tolerant to both pbp CSV schemas (with and without `pbp_id` / `wh_minutes`).

# 2026-03-15 - Primary multilayer point model bootstrapped

- Added `src/training/historical_point_dataset.py` to build a first supervised point-level dataset directly from Sackmann `tennis_pointbypoint` without requiring the full cached expansion first.
- Added `scripts/train_historical_point_model.py` and trained `artifacts/models/historical_point_model.joblib` on `100000` `t+2` rows (`logreg_pipeline`, `positive_rate=0.50326`).
- Added `src/live/point_model.py` with a layered point predictor that blends historical point classifier + Markov prior and optionally gates by execution-survival probability.
- Wired the layered predictor into `src/live/runtime.py`; point markets now automatically use the new supervised layer when `historical_point_model.joblib` exists.
- Verified test suite: `venv/bin/python -m unittest tests.test_historical_point_training tests.test_point_parser tests.test_live_point_training tests.test_live_runtime tests.test_config` -> `OK`.
- Dry-run over local snapshots showed the new point stack loads correctly, but several live markets are still skipped by `HistoricalLookup` because the current player-name coverage in the main historical match dataset is incomplete.

# 2026-03-15 - Name coverage expanded and model retrained

- Extended `HistoricalLookup` to index names not only from `player_match_stats`, but also directly from `atp_players.csv` / `wta_players.csv`, preserving namespace offsets.
- Added neutral-stat fallback for players that resolve by name but do not yet have local match history rows, so runtime no longer hard-fails immediately on those IDs.
- Added regression test covering CSV-based player resolution and neutral fallback stats.
- Retrained `historical_point_model.joblib` on a wider `t+2` corpus: `300000` rows, `1863` matches seen, still using a `logreg_pipeline`.
- Dry-run after retraining shows the multilayer point stack is live; remaining skipped markets are now concentrated in names that are absent even from the current player CSV sources and need another alias/source layer.

## 2026-03-15 Model Audit
- Audited current model artifacts and live logs in /opt/tennis_ai.
- Verified artifact loading under ./venv for all files in artifacts/models.
- Match winner models available and loadable:
  - lightgbm_elo.joblib -> sklearn.pipeline.Pipeline
  - logreg_elo.joblib -> sklearn.pipeline.Pipeline
  - lightgbm_baseline.joblib -> sklearn.pipeline.Pipeline
  - logreg_baseline.joblib -> sklearn.pipeline.Pipeline
- Reported ranking from artifacts/reports/model_metrics.json remains:
  - lightgbm_elo test_auc=0.742933 test_log_loss=0.593015
  - logreg_elo test_auc=0.735394 test_log_loss=0.600580
  - lightgbm_baseline test_auc=0.726878 test_log_loss=0.604859
  - logreg_baseline test_auc=0.706670 test_log_loss=0.622782
- Additional live/runtime artifacts verified:
  - historical_point_model.joblib -> sklearn.pipeline.Pipeline
  - historical_total_model.joblib -> sklearn.pipeline.Pipeline
  - game_markov_model.joblib -> src.live.markov.MarkovGameModel
  - point_outcome_model.joblib -> sklearn.dummy.DummyClassifier
  - leg_acceptance_model.joblib -> sklearn.dummy.DummyClassifier
  - execution_survival_model.joblib -> sklearn.dummy.DummyClassifier
- RL policy status:
  - artifacts/models/rl_bankroll_policy.json exists
  - model_type=bankroll_bandit_policy
  - no non-null rewards found in rl_outcomes.jsonl, so policy statistics are not meaningfully trained yet
- Live log summary at audit time:
  - rl_actions.jsonl rows=200, bets(real)=2, bet_fast_mode=1, bet_express=2, dry_run variants=13
  - rl_snapshots.jsonl rows=200, skipped=103, no_edge=63, placed=5
  - rl_outcomes.jsonl rows=26, pending=5, pending_dry_run=13, market_closed_unsettled=8, reward_non_null=0
- Main audit conclusion:
  - best current predictive model for match winner remains lightgbm_elo
  - RL/data logging is active, but RL reward/settlement is still missing
  - three live auxiliary models are currently placeholders (DummyClassifier), not production-grade learned models

## 2026-03-15 Batch Game Runner
- Added `scripts/run_live_tennis_game_batch.py` for one continuous process over all current live tennis root events.
- The batch runner discovers live events from `fonbet_tennis_live_events_latest`, then fetches each match through `ma/events/event`.
- Only `next_game_winner` markets are processed in this runner.
- Added pending-exposure control in `src/live/runtime.py`:
  - bankroll selection now uses `current_bankroll - pending_exposure`
  - new bets are blocked if stake would exceed the remaining available bankroll
- Current production command:
  - `./venv/bin/python scripts/run_live_tennis_game_batch.py --interval 5 --run-tag bankroll900_games --bankroll 900 --default-stake 30 --workers 6 --real`
- Current bankroll/staking profile:
  - global bankroll cap: `900`
  - allowed game stakes: `30 / 60 / 90`
  - market family: `next_game_winner` only
- First live observation from this runner:
  - batch logs are being written under `artifacts/live_betting/batch_bankroll900_games_*`
  - first processed live pool had `3` unique game markets across `3` events
  - no real bet has been placed yet in this run because all observed candidates ended as `no_bet` or `refresh_no_bet`

## 2026-03-16 RL Settlement Extension Audit
- Extended `settle_outcome_record()` in `src/live/runtime.py` so RL outcomes can now be settled for:
  - `set_total_over_under`
  - `point_plus_one_winner`
- The settlement path still leaves records as `market_closed_unsettled` when the winner cannot be inferred unambiguously from the final snapshot.
- Verified by manual runtime checks:
  - `set_total_over_under` example settled as `won`, `profit=27.0`, `reward=0.027`
  - `point_plus_one_winner` example settled as `won`, `profit=24.0`, `reward=0.024`
- Current artifact/log status after the patch:
  - `artifacts/live_betting/rl_outcomes.jsonl` mtime: `2026-03-16 19:21:30 +0100`
  - rows=`26`
  - status counts:
    - `pending_dry_run=13`
    - `pending=5`
    - `market_closed_unsettled=8`
  - market counts:
    - `match_winner=2`
    - `next_game_winner=2`
    - `point_plus_one_winner=2`
    - `set_total_over_under=20`
  - `reward_count=0`
- RL retrain activity recorded in `artifacts/live_betting/auto_live_event_betting.jsonl` on `2026-03-16`:
  - `13:15:46 UTC` retrain finished with `training_rows=580`
  - `15:34:30 UTC` retrain finished with `training_rows=812`
  - `16:04:32 UTC` retrain finished with `training_rows=1376`
  - `17:39:50 UTC` retrain finished with `training_rows=1613`
  - `17:45:49 UTC` retrain finished with `training_rows=1613`
  - `17:49:39 UTC` retrain finished with `training_rows=1640`
  - `18:19:57 UTC` retrain finished with `training_rows=3610`
- Manual RL policy export was run again:
  - command: `./venv/bin/python scripts/train_rl_policy.py`
  - output artifact: `artifacts/models/rl_bankroll_policy.json`
  - artifact mtime at audit: `2026-03-16 19:20:05 +0100`
  - `model_type=bankroll_bandit_policy`
  - `arms_count=0`
- Current conclusion:
  - the RL training loop now has settlement support for the live total and point markets it previously could not score
  - however, the existing `rl_outcomes.jsonl` still contains no historical `reward` rows, so the policy has not learned meaningful arm statistics yet
  - meaningful RL learning will start only after new live bets placed after this patch are later closed into `won/lost` records with non-null `reward`

## 2026-03-16 RL Outcome Backfill
- Added `scripts/backfill_rl_outcomes.py`.
- The script:
  - reads `artifacts/live_betting/rl_outcomes.jsonl`
  - reads `artifacts/live_betting/rl_tracker_state.json`
  - scans local snapshot sources including `*rl_snapshots.jsonl`, `market_snapshots.jsonl`, and `point_trajectories.jsonl`
  - attempts to re-run `settle_outcome_record()` for old rows with null `reward`
  - writes an audit trail to `artifacts/live_betting/rl_backfill_audit.jsonl`
  - creates timestamped backups before rewriting live artifacts
- Dry-run executed:
  - timestamp: `2026-03-16T18:28:27Z`
  - `rows_total=26`
  - `rows_backfilled=0`
  - `snapshot_sources_market_ids=306`
  - `snapshot_sources_event_ids=144`
- Real backfill executed:
  - timestamp: `2026-03-16T18:28:43Z`
  - `rows_total=26`
  - `rows_backfilled=0`
  - `tracker_pending_removed=0`
  - backup files created:
    - `artifacts/live_betting/rl_outcomes.jsonl.bak.20260316T182843Z`
    - `artifacts/live_betting/rl_tracker_state.json.bak.20260316T182843Z`
- Post-backfill state:
  - `artifacts/live_betting/rl_outcomes.jsonl` still has `reward_count=0`
  - rerunning `scripts/train_rl_policy.py` still produces `artifacts/models/rl_bankroll_policy.json` with `arms_count=0`
- Practical conclusion:
  - the backfill pipeline is now in place and can be rerun safely
  - the current local archive does not contain enough final live snapshots to resolve the old pending/unsettled rows into deterministic `won/lost`
  - new rows collected after the settlement patch are still the primary path to actual RL learning
