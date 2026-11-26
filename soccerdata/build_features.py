"""
Script 4: Construye el dataset de features para el modelo predictivo.

Lee los CSVs generados por los scrapers y crea un dataset final donde
cada fila es un partido con features calculadas a partir del historial
de cada equipo ANTES del partido (sin data leakage).

Genera: data/model_dataset.csv

Features:
  - Rolling stats (últimos 5 partidos): goles, puntos, racha, shots, etc.
  - Acumulado de temporada: posición estimada, puntos, gol diferencia
  - Historial head-to-head (últimos 5 enfrentamientos)
  - Odds de apuestas (implican probabilidades del mercado)
  - Temporales: día de la semana, mes
  - Variable objetivo: 'result' (H=local gana, D=empate, A=visitante gana)
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.environ.get("SOCCERDATA_DIR", "data")
ROLLING_WINDOW = 5


def load_matches() -> pd.DataFrame:
    """Carga y limpia el dataset de partidos combinado o de odds."""
    # Intentar cargar el combinado primero, si no existe usar odds
    for filename in ["matches_combined.csv", "matches_odds.csv"]:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            print(f"  Cargando {filename}...")
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError("No se encontraron archivos de partidos en data/")

    # Normalizar nombres de columnas según la fuente
    # Soporta tanto los nombres originales de football-data.co.uk (B365H, HS, etc.)
    # como los ya renombrados por scrape_matches.py (odds_home_b365, home_shots, etc.)
    col_map = {
        # Nombres originales de football-data.co.uk
        "Date": "date",
        "HomeTeam": "home",
        "AwayTeam": "away",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "result",
        "HTHG": "ht_home_goals",
        "HTAG": "ht_away_goals",
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_shots_target",
        "AST": "away_shots_target",
        "HF": "home_fouls",
        "AF": "away_fouls",
        "HC": "home_corners",
        "AC": "away_corners",
        "HY": "home_yellow",
        "AY": "away_yellow",
        "HR": "home_red",
        "AR": "away_red",
        "B365H": "odds_home",
        "B365D": "odds_draw",
        "B365A": "odds_away",
        # Nombres ya renombrados por scrape_matches.py (en matches_combined.csv)
        "odds_home_b365": "odds_home",
        "odds_draw_b365": "odds_draw",
        "odds_away_b365": "odds_away",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Fechas: primero se interpretan como ISO (YYYY-MM-DD, formato de matches_combined.csv).
    # Usar dayfirst=True directamente sobre fechas ISO invierte día y mes cuando el día es <= 12
    # (2017-08-11 -> 2017-11-08), lo que desordena el calendario y contamina las features rolling.
    # Solo las fechas que no son ISO (dd/mm/aaaa de football-data) se leen con dayfirst=True.
    fecha_iso = pd.to_datetime(df["date"], format="ISO8601", errors="coerce")
    fecha_dmy = pd.to_datetime(df["date"], format="mixed", dayfirst=True, errors="coerce")
    df["date"] = fecha_iso.fillna(fecha_dmy)
    df = df.dropna(subset=["date", "home", "away", "home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    # Calcular result si no existe
    if "result" not in df.columns:
        df["result"] = "D"
        df.loc[df["home_goals"] > df["away_goals"], "result"] = "H"
        df.loc[df["home_goals"] < df["away_goals"], "result"] = "A"

    # Control de calidad: cada pareja local-visitante se juega una vez por temporada.
    # Partidos repetidos distorsionan la forma reciente, los acumulados y el H2H.
    repetidos = int(df.duplicated(["season", "home", "away"]).sum())
    if repetidos:
        raise ValueError(
            f"{repetidos} partidos repetidos por (season, home, away). "
            "Regenerar matches_combined.csv con: python scrape_matches.py --solo-combinar"
        )

    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_team_events(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma partidos en eventos por equipo.
    Cada partido genera 2 filas: una para el local, otra para el visitante.
    """
    # Columnas de stats in-game (solo las que existen)
    stat_cols_home = {
        "home_shots": "shots",
        "home_shots_target": "shots_target",
        "home_corners": "corners",
        "home_fouls": "fouls",
        "home_yellow": "yellow_cards",
    }
    stat_cols_away = {
        "away_shots": "shots",
        "away_shots_target": "shots_target",
        "away_corners": "corners",
        "away_fouls": "fouls",
        "away_yellow": "yellow_cards",
    }

    # Eventos del equipo local
    home = pd.DataFrame({
        "date": matches["date"],
        "season": matches["season"],
        "team": matches["home"],
        "opponent": matches["away"],
        "is_home": 1,
        "goals_scored": matches["home_goals"],
        "goals_conceded": matches["away_goals"],
        "result": matches["result"],
    })
    home["points"] = home["result"].map({"H": 3, "D": 1, "A": 0})
    home["win"] = (home["result"] == "H").astype(int)
    home["loss"] = (home["result"] == "A").astype(int)

    # Eventos del equipo visitante
    away = pd.DataFrame({
        "date": matches["date"],
        "season": matches["season"],
        "team": matches["away"],
        "opponent": matches["home"],
        "is_home": 0,
        "goals_scored": matches["away_goals"],
        "goals_conceded": matches["home_goals"],
        "result": matches["result"],
    })
    away["points"] = away["result"].map({"H": 0, "D": 1, "A": 3})
    away["win"] = (away["result"] == "A").astype(int)
    away["loss"] = (away["result"] == "H").astype(int)

    # Agregar stats in-game si existen
    for src, dst in stat_cols_home.items():
        if src in matches.columns:
            home[dst] = matches[src].values
    for src, dst in stat_cols_away.items():
        if src in matches.columns:
            away[dst] = matches[src].values

    events = pd.concat([home, away]).sort_values("date").reset_index(drop=True)
    return events


def compute_rolling_features(events: pd.DataFrame) -> dict:
    """
    Calcula features rolling por equipo.
    Retorna dict[(date, team)] -> {feature_name: value}.
    Usa shift(1) para evitar data leakage (no incluye el partido actual).
    """
    rolling_cols = ["goals_scored", "goals_conceded", "points", "win", "loss"]
    # Agregar stats in-game si existen
    for col in ["shots", "shots_target", "corners"]:
        if col in events.columns:
            rolling_cols.append(col)

    features = {}
    for team, team_df in events.groupby("team"):
        team_df = team_df.sort_values("date").copy()

        # Rolling means (shifted para no incluir partido actual)
        for col in rolling_cols:
            team_df[f"roll_{col}"] = (
                team_df[col]
                .rolling(window=ROLLING_WINDOW, min_periods=1)
                .mean()
                .shift(1)
            )

        # Racha sin perder
        streak = []
        current = 0
        for pts in team_df["points"]:
            streak.append(current)
            current = current + 1 if pts > 0 else 0
        team_df["unbeaten_streak"] = streak

        # Acumulados de temporada (shifted)
        for season, sdf in team_df.groupby("season"):
            idx = sdf.index
            team_df.loc[idx, "season_matches"] = range(len(sdf))
            team_df.loc[idx, "season_points"] = sdf["points"].cumsum().shift(1).fillna(0)
            team_df.loc[idx, "season_goals_for"] = sdf["goals_scored"].cumsum().shift(1).fillna(0)
            team_df.loc[idx, "season_goals_against"] = sdf["goals_conceded"].cumsum().shift(1).fillna(0)
            team_df.loc[idx, "season_gd"] = (
                (sdf["goals_scored"] - sdf["goals_conceded"]).cumsum().shift(1).fillna(0)
            )
            team_df.loc[idx, "season_wins"] = sdf["win"].cumsum().shift(1).fillna(0)

        # Forma en casa vs fuera (últimos 5 en casa / fuera)
        home_mask = team_df["is_home"] == 1
        away_mask = team_df["is_home"] == 0
        team_df["roll_home_points"] = np.nan
        team_df["roll_away_points"] = np.nan
        if home_mask.sum() > 0:
            team_df.loc[home_mask, "roll_home_points"] = (
                team_df.loc[home_mask, "points"]
                .rolling(window=ROLLING_WINDOW, min_periods=1)
                .mean()
                .shift(1)
            )
        if away_mask.sum() > 0:
            team_df.loc[away_mask, "roll_away_points"] = (
                team_df.loc[away_mask, "points"]
                .rolling(window=ROLLING_WINDOW, min_periods=1)
                .mean()
                .shift(1)
            )
        # Forward fill para tener siempre un valor
        team_df["roll_home_points"] = team_df["roll_home_points"].ffill()
        team_df["roll_away_points"] = team_df["roll_away_points"].ffill()

        # Guardar features
        feat_cols = [c for c in team_df.columns if c.startswith(("roll_", "season_", "unbeaten"))]
        for _, row in team_df.iterrows():
            features[(row["date"], row["team"])] = {c: row[c] for c in feat_cols}

    return features


def compute_h2h_features(matches: pd.DataFrame) -> dict:
    """
    Historial head-to-head: últimos 5 enfrentamientos previos.
    Retorna dict[(date, home, away)] -> features.
    """
    features = {}
    matches = matches.sort_values("date").reset_index(drop=True)

    # Pre-agrupar enfrentamientos por par de equipos
    pair_matches = {}
    for idx, row in matches.iterrows():
        pair = tuple(sorted([row["home"], row["away"]]))
        if pair not in pair_matches:
            pair_matches[pair] = []
        pair_matches[pair].append(row)

    for idx, row in matches.iterrows():
        pair = tuple(sorted([row["home"], row["away"]]))
        home, away, date = row["home"], row["away"], row["date"]

        # Filtrar enfrentamientos previos
        prev = [m for m in pair_matches[pair] if m["date"] < date]
        prev = prev[-5:]  # Últimos 5

        if not prev:
            features[(date, home, away)] = {
                "h2h_home_wins": 0, "h2h_away_wins": 0,
                "h2h_draws": 0, "h2h_matches": 0,
                "h2h_home_goals_avg": 0, "h2h_away_goals_avg": 0,
            }
            continue

        hw, aw, dr = 0, 0, 0
        hg, ag = [], []
        for m in prev:
            if m["result"] == "D":
                dr += 1
            elif (m["home"] == home and m["result"] == "H") or \
                 (m["away"] == home and m["result"] == "A"):
                hw += 1
            else:
                aw += 1

            if m["home"] == home:
                hg.append(m["home_goals"])
                ag.append(m["away_goals"])
            else:
                hg.append(m["away_goals"])
                ag.append(m["home_goals"])

        features[(date, home, away)] = {
            "h2h_home_wins": hw,
            "h2h_away_wins": aw,
            "h2h_draws": dr,
            "h2h_matches": len(prev),
            "h2h_home_goals_avg": np.mean(hg),
            "h2h_away_goals_avg": np.mean(ag),
        }

    return features


def main():
    print("=" * 60)
    print("CONSTRUYENDO DATASET DE FEATURES")
    print("=" * 60)

    # 1. Cargar partidos
    print("\n[1] Cargando partidos...")
    matches = load_matches()
    print(f"    {len(matches)} partidos, {matches['season'].nunique()} temporadas")
    print(f"    Rango: {matches['date'].min()} a {matches['date'].max()}")

    # 2. Crear eventos por equipo
    print("\n[2] Creando eventos por equipo...")
    events = build_team_events(matches)
    print(f"    {len(events)} eventos, {events['team'].nunique()} equipos")

    # 3. Rolling features
    print("\n[3] Calculando rolling features...")
    rolling = compute_rolling_features(events)
    print(f"    {len(rolling)} registros")

    # 4. Head-to-head
    print("\n[4] Calculando head-to-head...")
    h2h = compute_h2h_features(matches)
    print(f"    {len(h2h)} pares")

    # 5. Ensamblar dataset
    print("\n[5] Ensamblando dataset...")
    rows = []
    for _, m in matches.iterrows():
        row = {
            "season": m["season"],
            "date": m["date"],
            "home": m["home"],
            "away": m["away"],
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
            "result": m["result"],
        }

        # Features temporales
        if pd.notna(m["date"]):
            row["day_of_week"] = m["date"].dayofweek
            row["month"] = m["date"].month
            row["is_weekend"] = int(m["date"].dayofweek >= 5)

        # Odds de apuestas (probabilidades implícitas)
        for odds_col in ["odds_home", "odds_draw", "odds_away"]:
            if odds_col in matches.columns and pd.notna(m.get(odds_col)):
                row[odds_col] = m[odds_col]
                row[f"{odds_col}_prob"] = 1.0 / m[odds_col]

        # Rolling features del equipo local
        home_feats = rolling.get((m["date"], m["home"]), {})
        for k, v in home_feats.items():
            row[f"home_{k}"] = v

        # Rolling features del equipo visitante
        away_feats = rolling.get((m["date"], m["away"]), {})
        for k, v in away_feats.items():
            row[f"away_{k}"] = v

        # Features diferenciales (home - away)
        diff_features = [
            "roll_goals_scored", "roll_points", "roll_win",
            "season_points", "season_gd", "season_wins",
        ]
        for feat in diff_features:
            hv = home_feats.get(feat)
            av = away_feats.get(feat)
            if hv is not None and av is not None:
                row[f"diff_{feat}"] = hv - av

        # H2H
        h2h_feats = h2h.get((m["date"], m["home"], m["away"]), {})
        row.update(h2h_feats)

        rows.append(row)

    dataset = pd.DataFrame(rows)

    # Eliminar filas sin rolling features (primeros partidos de cada equipo)
    before = len(dataset)
    dataset = dataset.dropna(subset=["home_roll_goals_scored", "away_roll_goals_scored"])
    print(f"    Eliminadas {before - len(dataset)} filas sin historial suficiente")

    # 6. Guardar
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "model_dataset.csv")
    dataset.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}")
    print(f"DATASET: {path}")
    print(f"  Filas:    {len(dataset)}")
    print(f"  Columnas: {len(dataset.columns)}")

    print(f"\nColumnas del dataset:")
    for i, col in enumerate(dataset.columns, 1):
        nulls = dataset[col].isna().sum()
        print(f"  {i:2d}. {col:<35s} (nulls: {nulls})")

    print(f"\nTarget 'result':")
    print(dataset["result"].value_counts().to_string())

    print(f"\nPróximo paso: entrena modelos con este dataset.")
    print(f"  Variable objetivo: 'result' (H/D/A)")
    print(f"  Excluye de features: season, date, home, away, home_goals, away_goals, result")


if __name__ == "__main__":
    main()
