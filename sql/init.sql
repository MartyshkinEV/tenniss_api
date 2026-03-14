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
