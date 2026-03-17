# Live Tennis Game Bet Notes

## Active Trading Rule
- Trade only `next_game_winner`.
- Do not trade `set_total_over_under`.
- Do not trade `point_plus_one_winner`.
- Any automation or manual live execution should stay game-only unless this rule is explicitly changed.

## Confirmed Procedure
1. Fetch `GET /ma/events/event?lang=en&version=...&eventId=...&scopeMarket=1600`.
2. Inspect `customFactors` for game markets.
3. Find the target game by:
   - `p=100`, `pt="1"` for game 1
   - `p=300`, `pt="3"` for game 3
   - `p=400`, `pt="4"` for game 4
   - `p=500`, `pt="5"` for game 5
   - `p=600`, `pt="6"` for game 6
   - `p=700`, `pt="7"` for game 7
4. Extract both candidate `factorId` values for that game from `customFactors`.
5. Confirm the selected factor through `POST /coupon/betSlipInfo`.
6. Use the accepted selection in the request chain:
   - `POST /coupon/betRequestId`
   - `POST /coupon/bet`
   - `POST /coupon/betResult`

## Working Bet Flow
1. Find a live tennis event with an open game market.
2. Read the subevent for the current set from `ma/events/event`.
3. Pick the exact game market by `factorId + param`:
   - example: `1754 + param 300` -> `Game 3 - player2`
   - example: `1753 + param 300` -> `Game 3 - player1`
4. Send `betSlipInfo` with:
   - `lang`
   - `fsid`
   - `sysId`
   - `clientId`
   - `CDI`
   - `deviceId`
   - `scopeMarketId`
   - `bets: [{eventId, factorId, param, old: true}]`
5. Only continue if `betSlipInfo` returns:
   - non-zero `factor.v`
   - current `event.score`
   - valid limits in `sums.min` / `sums.max`
6. Request a fresh `requestId` through `POST /coupon/betRequestId`.
7. Send `POST /coupon/bet` with one coupon bet:
   - `event`
   - `factor`
   - `value`
   - `param`
   - `score`
   - `zone: "es"`
8. Finish with `POST /coupon/betResult` using the same `requestId`.
9. Save the accepted registration fields:
   - `requestId`
   - `regId`
   - `checkCode`
   - final odds
   - final score snapshot

## ML Flow
1. Take the live root `eventId`.
2. Load `ma/events/event` and extract all `next_game_winner` markets from the current set subevent.
3. Build the live feature frame for the event state:
   - player ids
   - surface / level / round
   - live score and serving side
   - historical hold/break features
4. Score each game market with the ML runtime:
   - historical game model
   - Markov next-game model
   - combined `player1_probability`
   - combined `player2_probability`
5. Convert probabilities into candidate selections:
   - implied probability from current odds
   - `edge = model_probability - implied_probability`
   - acceptance probability
   - ranking score
6. Keep only the best candidate for the event, or restrict to a chosen game with `--target-game`.
7. Before placing the bet, refresh the exact selection through `betSlipInfo`.
8. Only place the bet if refresh returns a live non-zero `factor.v`.
9. Send `betRequestId -> bet -> betResult`.

## ML Command
Run ML analysis and print the ready request chain without placing a bet:
```bash
./venv/bin/python scripts/prepare_game_bet_request.py --event-id <EVENT_ID>
```

Restrict the search to a specific game:
```bash
./venv/bin/python scripts/prepare_game_bet_request.py --event-id <EVENT_ID> --target-game 3
```

Wait for a valid ML candidate to appear:
```bash
./venv/bin/python scripts/prepare_game_bet_request.py --event-id <EVENT_ID> --wait-open-seconds 30 --poll-interval 1
```

Run ML, refresh the selection, and place the real bet:
```bash
./venv/bin/python scripts/prepare_game_bet_request.py --event-id <EVENT_ID> --target-game 3 --send-bet
```

Force a side after ML filtering:
```bash
./venv/bin/python scripts/prepare_game_bet_request.py --event-id <EVENT_ID> --target-game 3 --side player2 --send-bet
```

## State-Only ML Flow
Use this when the full runtime is too heavy or too slow for live execution.

What it uses:
- `historical_game_model.joblib`
- `MarkovGameModel`
- live state only: current game score, point score, serve side, target game

What it skips:
- full player-history lookup
- point model
- total model
- acceptance model

Command:
```bash
./venv/bin/python scripts/place_ml_game_bet_state_only.py --event-id <EVENT_ID> --send-bet
```

Optional target game filter:
```bash
./venv/bin/python scripts/place_ml_game_bet_state_only.py --event-id <EVENT_ID> --target-game 5 --send-bet
```

Decision rule:
1. Load live `next_game_winner` markets from `ma/events/event`.
2. Score each game market with:
   - historical game model probability
   - Markov next-game probability
   - weighted blend
3. Convert odds into implied probability.
4. Compute `edge = model_probability - implied_probability`.
5. Keep candidates that pass runtime thresholds.
6. Refresh the selected candidate through `betSlipInfo`.
7. Place the bet only if refresh returns live non-zero `factor.v`.

## Important Rules
- Do not discover game bets from `betSlipInfo` alone.
- First locate the market in `events/event`, then use `betSlipInfo` only for validation.
- For live bets, keep a single selection in the coupon.
- Use the live zone expected by the confirmed payload path for the current flow.

## Data To Record Per Bet
- `eventId`
- match players
- current live score
- target game number
- market markers: `p`, `pt`
- both candidate `factorId` values
- which side/player each `factorId` maps to
- displayed odds/value before placement
- confirmed odds/value from `betSlipInfo`
- minimum stake
- stake amount
- `requestId`
- `regId`
- `checkCode`
- final result payload path if saved

## Confirmed Examples
- `63319046`, game 5, `p=500`
  - `1747` -> `Game 5 - Smith Alana`
  - `1748` -> `Game 5 - Bartashevich Y`
- `63319501`, game 4, `p=400`
  - `9961` -> `Game 4 - Juan Mano A`
  - `9962` -> `Game 4 - Arnaboldi F`
- `63319501`, game 6, `p=600`
  - `1750` -> `Game 6 - Juan Mano A`
  - `1751` -> `Game 6 - Arnaboldi F`
- `63299287`, game 1, `p=100`
  - `1747` -> `Game 1 - Borrelli L`
  - `1748` -> `Game 1 - Wessels L`

## Confirmed Successful Bet Samples
- `63337687`
  - match: `Pieri S` vs `Campana Lee G`
  - selection: `Game 3 - Campana Lee G`
  - `factorId=1754`
  - `param=300`
  - accepted odds: `1.27`
  - stake: `30`
  - `requestId=12340724104`
  - `regId=5411733406`
  - `checkCode=VX-X2-M6`
- `63337695`
  - match: `Lokoli L` vs `Kirkin E`
  - selection: `Game 3 - Lokoli L`
  - `factorId=1753`
  - `param=300`
  - accepted odds: `1.60`
  - stake: `30`
  - `requestId=12340733242`
  - `regId=5411813013`
  - `checkCode=N3-CP-HX`
- `63339432`
  - match: `Yevseyev D` vs `Pieczonka F`
  - selection: `Game 5 - Pieczonka F`
  - `factorId=1748`
  - `param=500`
  - accepted odds: `2.45`
  - ML mode: `state-only`
  - model probability: `0.5574`
  - edge: `0.1492`
  - stake: `30`
  - `requestId=12340862005`
  - `regId=5412844323`
  - `checkCode=Z3-3N-C2`
- `63340264`
  - match: `Miyazaki Y L` vs `Ryser V`
  - selection: `Game 7 - Miyazaki Y L`
  - `factorId=1753`
  - `param=700`
  - accepted odds: `1.60`
  - ML mode: `state-only`
  - model probability: `0.7410`
  - edge: `0.1160`
  - stake: `30`
  - `requestId=12340862360`
  - `regId=5412845665`
  - `checkCode=19-6H-SR`
- `63326892`
  - match: `Gill F` vs `Arnaboldi F`
  - selection: `Game 3 - Arnaboldi F`
  - `factorId=1754`
  - `param=300`
  - accepted odds: `1.50`
  - ML mode: `state-only`
  - model probability: `0.7522`
  - edge: `0.0856`
  - stake: `30`
  - `requestId=12340878158`
  - `regId=5412974052`
  - `checkCode=VQ-18-WP`
- `63339750`
  - match: `Ebster A-L` vs `Gaillard D`
  - selection: `Game 10 - Gaillard D`
  - `factorId=1751`
  - `param=1000`
  - accepted odds: `1.33`
  - ML mode: `state-only`
  - model probability: `0.8151`
  - edge: `0.0632`
  - stake: `30`
  - `requestId=12340878413`
  - `regId=5412976143`
  - `checkCode=8G-5W-S2`
- `63340578`
  - match: `Svatikova L` vs `Adams J`
  - selection: `Game 2 - Adams J`
  - `factorId=1751`
  - `param=200`
  - accepted odds: `1.85`
  - ML mode: `state-only`
  - model probability: `0.7472`
  - edge: `0.2066`
  - stake: `30`
  - `requestId=12340893417`
  - `regId=5413097782`
  - `checkCode=NH-6H-3V`

## Example Payloads
`betSlipInfo`:
```json
{
  "lang": "en",
  "fsid": "<fsid>",
  "sysId": 21,
  "clientId": 20211191,
  "CDI": 0,
  "deviceId": "<deviceId>",
  "scopeMarketId": "1600",
  "bets": [
    {
      "eventId": 63337695,
      "factorId": 1753,
      "param": 300,
      "old": true
    }
  ]
}
```

`bet`:
```json
{
  "requestId": 12340733242,
  "lang": "en",
  "fsid": "<fsid>",
  "sysId": 21,
  "clientId": 20211191,
  "coupon": {
    "amount": 30,
    "flexBet": "any",
    "flexParam": true,
    "mirror": "https://fon.bet",
    "bets": [
      {
        "num": 1,
        "event": 63337695,
        "factor": 1753,
        "value": 1.60,
        "param": 300,
        "score": "2:0",
        "zone": "es"
      }
    ]
  }
}
```
