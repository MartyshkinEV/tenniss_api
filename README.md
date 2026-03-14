# tenniss_api

ML/SQL-пайплайн для прогнозирования теннисных матчей (ATP) на PostgreSQL.

## Что есть сейчас
- загрузка ATP-данных в PostgreSQL;
- построение табличных фичей (`match_features`, `match_features_elo`);
- обучение baseline моделей (logreg/lightgbm);
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
python scripts/train_match_model.py
python scripts/train_lightgbm_model.py
python scripts/train_lightgbm_elo_model.py
```

8. Сделать batch-predict:
```bash
python scripts/predict_match_model.py
```

9. Сделать point-predict для пары игроков:
```bash
python scripts/predict_match.py <P1_ID> <P2_ID>
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
- `scripts/build_match_features.py` — формирование `match_features`.
- `scripts/build_elo_features.py` — добавление ELO-фичей (`match_features_elo`).
- `scripts/train_match_model.py` — обучение logistic baseline.
- `scripts/train_lightgbm_model.py` — обучение lightgbm baseline.
- `scripts/train_lightgbm_elo_model.py` — обучение lightgbm с ELO.
- `scripts/predict_match_model.py` — пакетный предикт в `match_predictions`.
- `scripts/predict_match.py` — предикт для конкретной пары игроков.

## Что пока осталось в `scripts/`
Это намеренно: текущая логика оставлена с thin-refactor для сохранения работоспособности пайплайна.
Новая архитектурная точка расширения добавлена в `src/` (models/features/training/inference).

## TODO следующей итерации
- Единый dataset builder и единый train/eval pipeline в `src/`.
- Time-based split как общий компонент.
- Подключение xgboost/catboost через опциональные зависимости.
- Модельный registry/metadata.
- Поддержка WTA.
