# Current Session

## Goal
Stabilize the current refactor of the tennis prediction pipeline and finish the new live betting runtime flow without losing the older training/prediction workflow.

## Current State
- Large local worktree is not committed yet.
- Core configuration has been centralized in `config.py`.
- Training and prediction scripts were moved toward a shared config/model registry approach.
- A new live betting runtime was added under `src/live/`.
- README already reflects the newer architecture and commands.
- Current live trading directive:
  - play only `next_game_winner`
  - do not place `set_total_over_under`
  - do not place `point_plus_one_winner`

## Done
- Added centralized settings loading from `.env` in `config.py`.
- Added model metadata and default production model resolution.
- Updated training scripts to use shared config and artifact locations.
- Updated prediction scripts to use the resolved default production artifact.
- Added model comparison script: `scripts/compare_models.py`.
- Added live runtime entrypoint: `scripts/run_live_betting.py`.
- Added next-game support:
  - Fonbet parser now recognizes `next_game_winner` markets from factors `1750/1751`
  - Markov next-game model in `src/live/markov.py`
  - bankroll-aware RL policy in `src/live/policy.py`
- Added live runtime/domain code in `src/live/runtime.py` and `src/live/fonbet.py`.
- Added tests for config, feature builder, prediction flow, and live runtime.

## In Progress
- Finish integrating live runtime with the rest of the project.
- Replace generic Fonbet payload templates with the exact confirmed request shape from captured browser traffic.
- Verify that new scripts and tests all pass together in the current environment.
- Remove remaining duplication between `scripts/predict_match.py` and live runtime feature assembly.

## Likely Open Questions
- Whether live betting should remain dry-run oriented by default or support real placement immediately after validation.
- Whether `scripts/predict_match.py` should be further reduced and moved fully into `src/`.
- Whether the production default should always remain `lightgbm_elo.joblib` or be derived from the comparison report.
- Which Fonbet fields are mandatory for real bet placement beyond the already confirmed `fsid`, `clientId`, `sysId`, `deviceId`, `requestId`, and `coupon.bets[*]`.

## Confirmed Fonbet Notes
- Confirmed endpoint chain:
  - `POST /coupon/betSlipInfo`
  - `POST /coupon/bet`
  - `POST /coupon/betResult`
- Confirmed browser-like headers are required:
  - `Content-Type: text/plain;charset=UTF-8`
  - `Origin`
  - `Referer`
  - `User-Agent`
  - `Cookie`
  - optional `Authorization: Bearer ...`
- Confirmed request metadata fields from captured traffic:
  - `lang`
  - `fsid`
  - `sysId`
  - `clientId`
  - `CDI`
  - `deviceId`
  - `requestId` for bet/betResult flow
- Confirmed coupon structure includes:
  - `amount`
  - `flexBet`
  - `flexParam`
  - `mirror`
  - `bets`
- Confirmed each bet item may include:
  - `num`
  - `event`
  - `factor`
  - `value`
  - `score` for live state
  - `zone`
- Confirmed `zone: "lv"` means live and `zone: "sp"` means prematch.
- Confirmed Fonbet rejects mixed live and prematch bets in one coupon.

## Important Files
- `config.py`
- `README.md`
- `scripts/predict_match.py`
- `scripts/predict_match_model.py`
- `scripts/compare_models.py`
- `scripts/run_live_betting.py`
- `src/features/feature_builder.py`
- `src/live/runtime.py`
- `src/live/fonbet.py`
- `tests/test_live_runtime.py`

## Fast Resume Checklist
1. Read this file.
2. Read `docs/NEXT_STEPS.md`.
3. Read the latest entry in `docs/WORKLOG.md`.
4. Run `git status --short`.
5. If needed, run targeted tests for the files being changed.
