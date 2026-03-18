CREATE TABLE IF NOT EXISTS players (
    player_id BIGINT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    hand TEXT,
    birth_date DATE,
    country_code TEXT,
    height_cm INTEGER
);

CREATE TABLE IF NOT EXISTS rankings (
    ranking_date DATE NOT NULL,
    ranking INTEGER NOT NULL,
    player_id BIGINT NOT NULL,
    ranking_points INTEGER,
    PRIMARY KEY (ranking_date, player_id)
);

CREATE INDEX IF NOT EXISTS idx_rankings_player_date
ON rankings(player_id, ranking_date DESC);

CREATE TABLE IF NOT EXISTS matches (
    source_file TEXT,
    level_group TEXT,
    tourney_id TEXT,
    tourney_name TEXT,
    surface TEXT,
    draw_size INTEGER,
    tourney_level TEXT,
    tourney_date DATE,
    match_num INTEGER,
    best_of INTEGER,
    round TEXT,
    minutes INTEGER,

    winner_id BIGINT,
    winner_name TEXT,
    winner_hand TEXT,
    winner_ht INTEGER,
    winner_ioc TEXT,
    winner_age NUMERIC,

    loser_id BIGINT,
    loser_name TEXT,
    loser_hand TEXT,
    loser_ht INTEGER,
    loser_ioc TEXT,
    loser_age NUMERIC,

    winner_rank INTEGER,
    winner_rank_points INTEGER,
    loser_rank INTEGER,
    loser_rank_points INTEGER,

    score TEXT,

    w_ace INTEGER, w_df INTEGER, w_svpt INTEGER, w_1stin INTEGER, w_1stwon INTEGER, w_2ndwon INTEGER,
    w_svgms INTEGER, w_bpsaved INTEGER, w_bpfaced INTEGER,
    l_ace INTEGER, l_df INTEGER, l_svpt INTEGER, l_1stin INTEGER, l_1stwon INTEGER, l_2ndwon INTEGER,
    l_svgms INTEGER, l_bpsaved INTEGER, l_bpfaced INTEGER
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(tourney_date);
CREATE INDEX IF NOT EXISTS idx_matches_winner_date ON matches(winner_id, tourney_date DESC);
CREATE INDEX IF NOT EXISTS idx_matches_loser_date ON matches(loser_id, tourney_date DESC);
CREATE INDEX IF NOT EXISTS idx_matches_surface_date ON matches(surface, tourney_date DESC);

CREATE TABLE IF NOT EXISTS pointbypoint (
    pbp_id TEXT,
    source_file TEXT,
    match_date DATE,
    tny_name TEXT,
    tour TEXT,
    draw TEXT,
    server1 TEXT,
    server2 TEXT,
    winner INTEGER,
    match_score TEXT,
    adf_flag INTEGER,
    wh_minutes INTEGER,
    set_no INTEGER NOT NULL,
    game_no INTEGER NOT NULL,
    point_no INTEGER NOT NULL,
    is_tiebreak BOOLEAN NOT NULL,
    server_name TEXT,
    returner_name TEXT,
    point_winner_name TEXT,
    point_winner_side INTEGER,
    score_before_p1 TEXT,
    score_before_p2 TEXT,
    score_after_p1 TEXT,
    score_after_p2 TEXT,
    point_code TEXT,
    PRIMARY KEY (pbp_id, set_no, game_no, point_no)
);

CREATE INDEX IF NOT EXISTS idx_pointbypoint_match_date ON pointbypoint(match_date);
CREATE INDEX IF NOT EXISTS idx_pointbypoint_pbp_id ON pointbypoint(pbp_id);
CREATE INDEX IF NOT EXISTS idx_pointbypoint_tour ON pointbypoint(tour);

CREATE TABLE IF NOT EXISTS live_market_snapshots (
    market_id TEXT PRIMARY KEY,
    timestamp_utc TIMESTAMPTZ,
    event_id TEXT NOT NULL,
    competition TEXT,
    surface TEXT,
    round_name TEXT,
    best_of INTEGER,
    tourney_level TEXT,
    player1_name TEXT,
    player2_name TEXT,
    player1_odds DOUBLE PRECISION,
    player2_odds DOUBLE PRECISION,
    market_type TEXT,
    live_score TEXT,
    live_comment TEXT,
    live_delay INTEGER,
    serving_team INTEGER,
    player1_factor_id INTEGER,
    player2_factor_id INTEGER,
    player1_param TEXT,
    player2_param TEXT,
    scope_market_id TEXT,
    target_game_number INTEGER,
    target_point_number INTEGER,
    zone TEXT
);

CREATE INDEX IF NOT EXISTS idx_live_market_snapshots_event_id ON live_market_snapshots(event_id);
CREATE INDEX IF NOT EXISTS idx_live_market_snapshots_market_type ON live_market_snapshots(market_type);

CREATE TABLE IF NOT EXISTS bet_log (
    bet_id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    settled_at TIMESTAMPTZ,
    match_id TEXT NOT NULL,
    event_id TEXT,
    market TEXT NOT NULL,
    market_type TEXT,
    pick TEXT NOT NULL,
    odds DOUBLE PRECISION NOT NULL,
    stake DOUBLE PRECISION NOT NULL,
    result TEXT NOT NULL DEFAULT 'pending',
    profit DOUBLE PRECISION,
    model_prob DOUBLE PRECISION NOT NULL,
    bookmaker_prob DOUBLE PRECISION NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION,
    threshold_value DOUBLE PRECISION,
    min_probability DOUBLE PRECISION,
    data_quality_score DOUBLE PRECISION,
    filter_surface TEXT,
    filter_tourney_level TEXT,
    filter_form_window INTEGER,
    filter_passed BOOLEAN NOT NULL DEFAULT FALSE,
    decision_reason TEXT NOT NULL,
    explanation_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_bet_log_match_market
ON bet_log(match_id, market, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_bet_log_market_result
ON bet_log(market, result, created_at DESC);

CREATE TABLE IF NOT EXISTS odds_history (
    odds_history_id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    match_id TEXT NOT NULL,
    event_id TEXT,
    market TEXT NOT NULL,
    market_id TEXT,
    selection TEXT,
    line_value DOUBLE PRECISION,
    odds DOUBLE PRECISION NOT NULL,
    bookmaker_prob DOUBLE PRECISION,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL DEFAULT 'live_market_snapshots',
    raw_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_odds_history_match_market_ts
ON odds_history(match_id, market, timestamp_utc DESC);

CREATE INDEX IF NOT EXISTS idx_odds_history_event_market_ts
ON odds_history(event_id, market, timestamp_utc DESC);

CREATE TABLE IF NOT EXISTS game_stats (
    player_id BIGINT PRIMARY KEY,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    avg_games DOUBLE PRECISION,
    hold_pct DOUBLE PRECISION,
    break_pct DOUBLE PRECISION,
    tiebreak_rate DOUBLE PRECISION,
    "3set_rate" DOUBLE PRECISION,
    sample_matches INTEGER NOT NULL DEFAULT 0,
    stats_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS live_match_player_stats (
    source TEXT NOT NULL,
    event_id TEXT NOT NULL,
    snapshot_utc TIMESTAMPTZ NOT NULL,
    requested_url TEXT,
    resolved_url TEXT,
    player1_name TEXT,
    player2_name TEXT,
    candidate_count INTEGER,
    status TEXT NOT NULL DEFAULT 'ok',
    message TEXT,
    stats_json JSONB NOT NULL,
    normalized_player_stats_json JSONB NOT NULL,
    PRIMARY KEY (source, event_id, snapshot_utc)
);

CREATE INDEX IF NOT EXISTS idx_live_match_player_stats_event_id
ON live_match_player_stats(event_id);

CREATE INDEX IF NOT EXISTS idx_live_match_player_stats_snapshot_utc
ON live_match_player_stats(snapshot_utc DESC);

CREATE TABLE IF NOT EXISTS fonbet_event_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    snapshot_utc TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    requested_url TEXT NOT NULL,
    response_headers_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload_json JSONB NOT NULL,
    sports_count INTEGER NOT NULL DEFAULT 0,
    events_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_fonbet_event_snapshots_snapshot_utc
ON fonbet_event_snapshots(snapshot_utc DESC);

CREATE TABLE IF NOT EXISTS fonbet_events (
    snapshot_id BIGINT NOT NULL REFERENCES fonbet_event_snapshots(snapshot_id) ON DELETE CASCADE,
    snapshot_utc TIMESTAMPTZ NOT NULL,
    event_id BIGINT NOT NULL,
    parent_id BIGINT,
    sport_id INTEGER NOT NULL,
    sport_name TEXT,
    sport_alias TEXT,
    sport_kind TEXT,
    root_sport_id INTEGER,
    root_sport_name TEXT,
    event_name TEXT,
    level INTEGER,
    place TEXT,
    team1 TEXT,
    team2 TEXT,
    priority INTEGER,
    sort_order TEXT,
    raw_event_json JSONB NOT NULL,
    PRIMARY KEY (snapshot_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_fonbet_events_snapshot_utc
ON fonbet_events(snapshot_utc DESC);

CREATE INDEX IF NOT EXISTS idx_fonbet_events_sport_place
ON fonbet_events(sport_id, place, snapshot_utc DESC);

CREATE INDEX IF NOT EXISTS idx_fonbet_events_root_sport_place
ON fonbet_events(root_sport_id, place, snapshot_utc DESC);

CREATE INDEX IF NOT EXISTS idx_fonbet_events_event_id
ON fonbet_events(event_id, snapshot_utc DESC);

CREATE OR REPLACE VIEW fonbet_tennis_events_latest AS
SELECT *
FROM fonbet_events
WHERE snapshot_id = (SELECT MAX(snapshot_id) FROM fonbet_event_snapshots)
  AND root_sport_id = 4;

CREATE OR REPLACE VIEW fonbet_tennis_live_events_latest AS
SELECT *
FROM fonbet_events
WHERE snapshot_id = (SELECT MAX(snapshot_id) FROM fonbet_event_snapshots)
  AND root_sport_id = 4
  AND place = 'live';
