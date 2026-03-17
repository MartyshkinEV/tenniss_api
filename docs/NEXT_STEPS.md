# Next Steps

1. Update `src/live/fonbet.py` to build exact single live coupon payloads using the confirmed Fonbet fields.
2. Keep coupon generation restricted to one live tennis selection with `zone: "lv"` until real successful bet samples are captured.
3. Run the relevant test suite and record which parts still fail.
4. Verify `scripts/run_live_betting.py --once` in dry-run mode against a local feed file.

## Current Priority

1. Fill `.env` with live Fonbet session data.
2. Run a real `next_game_winner` dry-run against the current Fonbet feed.
3. When a suitable market is found, place the first real next-game bet through the new `param + zone="es"` payload path.
4. Connect a real settlement source so RL outcomes get numeric `reward` and `pnl`.
5. Refactor duplicated prediction feature-building logic into shared code under `src/`.
6. Confirm artifact resolution behavior between `artifacts/models/` and legacy `models/match_winner/`.
7. Regenerate the model comparison report after tests and local artifact verification.

## Useful Commands
```bash
git status --short
python -m pytest tests/test_config.py tests/test_feature_builder.py tests/test_predict_match.py tests/test_live_runtime.py
./venv/bin/python scripts/compare_models.py --write-report
./venv/bin/python scripts/run_live_betting.py --once --feed-file <path-to-json>
```

## Risks To Check
- Tests may rely on local artifacts or dataset availability.
- Live provider payloads may still vary from the normalized assumptions in `src/live/fonbet.py`.
- `betSlipInfo` and `betResult` full successful payloads/responses are still incomplete in notes.
- Some script logic is duplicated between ad hoc CLI scripts and `src/` modules.
