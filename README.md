# tenniss_api

ML/SQL-пайплайн для прогнозирования теннисных матчей (ATP) на PostgreSQL.

## Что есть сейчас
- загрузка ATP-данных в PostgreSQL;
- построение табличных фичей (`match_features`, `match_features_elo`);
- обучение baseline моделей (logreg/lightgbm);
- отдельные market-модели для `match_winner`, `games_total`, `games_handicap`, `three_sets`;
- инференс и запись предсказаний в таблицу.

## Структура проекта
```text
tenniss_api/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config.py
├── scripts/
├── sql/
├── src/
│   ├── db/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── inference/
│   └── utils/
├── tests/
└── artifacts/
```

## Требования
- Python 3.10+
- PostgreSQL

## Быстрый старт
1. Создать окружение:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Установить зависимости:
```bash
pip install -r requirements.txt
```

3. Создать `.env`:
```bash
cp .env.example .env
```

4. Инициализировать БД:
```bash
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f sql/init.sql
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f sql/player_match_stats.sql
```

5. Загрузить ATP-данные (CSV должны лежать в `DATA_DIR`):
```bash
python scripts/load_atp.py
```

6. Построить фичи:
```bash
python scripts/build_match_features.py
python scripts/build_elo_features.py
```

7. Обучить модель:
```bash
./venv/bin/python scripts/train_match_model.py
./venv/bin/python scripts/train_lightgbm_model.py
./venv/bin/python scripts/train_logreg_elo_model.py
./venv/bin/python scripts/train_lightgbm_elo_model.py
./venv/bin/python scripts/train_market_models.py
```

8. Сделать batch-predict:
```bash
./venv/bin/python scripts/predict_match_model.py
```

9. Сделать point-predict для пары игроков:
```bash
./venv/bin/python scripts/predict_match.py <P1_ID> <P2_ID>
```

10. Сравнить все обученные модели и обновить отчет:
```bash
./venv/bin/python scripts/compare_models.py --write-report
```

## Конфигурация
Все пути и подключение к БД задаются через `config.py` + переменные окружения.

Ключевые переменные:
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `PROJECT_ROOT`, `DATA_DIR`, `ARTIFACTS_DIR`, `MODELS_DIR`, `SQL_DIR`
- `LOG_LEVEL`

## Где лежат артефакты
- Основной путь: `artifacts/models/` (через `MODELS_DIR`).
- Для обратной совместимости в части скриптов оставлен fallback чтения из `models/match_winner/`.

## Назначение текущих скриптов
- `scripts/load_atp.py` — загрузка players/rankings/matches.
- `scripts/load_matches_only.py` — загрузка только matches.
- `scripts/ingest_fonbet_events.py` — каждые 20 секунд сохраняет полный `ma/events/list` в `fonbet_event_snapshots` и раскладывает события в `fonbet_events`.
- `scripts/build_match_features.py` — формирование `match_features`.
- `scripts/build_elo_features.py` — добавление ELO-фичей (`match_features_elo`).
- `scripts/train_match_model.py` — обучение logistic baseline.
- `scripts/train_lightgbm_model.py` — обучение lightgbm baseline.
- `scripts/train_lightgbm_elo_model.py` — обучение lightgbm с ELO.
- `scripts/train_historical_point_model.py` — обучение исторической модели `кто выиграет точку через offset`.
- `scripts/train_historical_game_model.py` — обучение исторической модели `кто выиграет текущий гейм по состоянию поинта`.
- `scripts/train_market_models.py` — обучение отдельных моделей по рынкам `match_winner / games_total / games_handicap / three_sets` в вариантах `catboost` и `logreg`.
- `scripts/run_market_backtest.py` — backtest и ROI/calibration-отчет по журналу ставок или CSV предсказаний.
- `scripts/predict_match_model.py` — пакетный предикт; production default это `lightgbm_elo.joblib`.
- `scripts/predict_match.py` — point-predict для конкретной пары игроков через production default model.
- `scripts/compare_models.py` — сравнение всех локально доступных обученных моделей и генерация отчета.

## Fonbet events/list
Инициализация новых таблиц:
```bash
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f sql/init.sql
```

Запуск непрерывной загрузки полного каталога:
```bash
python scripts/ingest_fonbet_events.py --url "https://line-lb61-w.bk6bba-resources.com/ma/events/list?lang=en&version=72533210346&scopeMarket=1600"
```

Нормализация live-маркетов теперь может идти без второго HTTP-таймера, из последнего raw-снапшота в БД:
```bash
python scripts/ingest_live_markets.py --provider events
```

Прямой режим `--provider fonbet` оставлен только как fallback и для обычного потока больше не нужен.

Асинхронный per-event ML watcher:
```bash
python scripts/run_event_ml_watch.py
```

Что делает watcher:
- берёт актуальные live tennis match `event_id` из `fonbet_tennis_live_events_latest`
- поднимает отдельную async-задачу на каждый event
- циклически тянет `ma/events/event?eventId=...`
- прогоняет ML scoring по доступным рынкам
- пишет лучшие кандидаты в `artifacts/live_betting/event_ml_watch.jsonl`

Одноразовый прогон по конкретному событию:
```bash
python scripts/run_event_ml_watch.py --event-id 63295583 --once
```

Непрерывный runtime по одному событию с полным RL-логированием:
```bash
python scripts/run_event_live_betting.py --event-id 63295863 --market-type all
```

Что даёт single-event runtime:
- непрерывно тянет только один `eventId` через `ma/events/event`
- использует существующий `LiveBettingRuntime`
- Марковская модель + ML модели скорят рынки в каждом цикле
- RL policy решает `bet / no_bet / duplicate / refresh_no_bet`
- пишет event-specific файлы:
  - `*_decisions.jsonl`
  - `*_rl_snapshots.jsonl`
  - `*_rl_actions.jsonl`
  - `*_rl_outcomes.jsonl`
  - `*_point_trajectories.jsonl`

Полезные выборки:
```sql
SELECT snapshot_id, snapshot_utc, events_count
FROM fonbet_event_snapshots
ORDER BY snapshot_utc DESC;

SELECT *
FROM fonbet_events
WHERE root_sport_id = 4
ORDER BY snapshot_utc DESC, event_id;

SELECT *
FROM fonbet_events
WHERE root_sport_id = 4 AND place = 'live'
ORDER BY snapshot_utc DESC, event_id;

SELECT * FROM fonbet_tennis_events_latest;
SELECT * FROM fonbet_tennis_live_events_latest;
```

## Что пока осталось в `scripts/`
Это намеренно: текущая логика оставлена с thin-refactor для сохранения работоспособности пайплайна.
Новая архитектурная точка расширения добавлена в `src/` (models/features/training/inference).

## TODO следующей итерации
- Единый dataset builder и единый train/eval pipeline в `src/`.
- Time-based split как общий компонент.
- Модельный registry/metadata.
- Поддержка WTA.

## Betting analytics
- `sql/init.sql` теперь создает `bet_log`, `odds_history` и `game_stats`.
- live snapshot recorder пишет историю линии в `odds_history`.
- runtime пишет причину решения и value-метрики в `artifacts/betting/bet_log.jsonl`.
- backtest/report:
```bash
python scripts/run_market_backtest.py --input artifacts/betting/bet_log.jsonl
```
