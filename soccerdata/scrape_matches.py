"""
Script 1: Extrae resultados de partidos de la Premier League.

Usa soccerdata (FBref) para fixtures + xG, y football-data.co.uk para
estadísticas de apuestas (odds).

Genera:
  data/matches_fbref.csv       - Partidos con xG de FBref
  data/matches_odds.csv        - Partidos con odds de football-data.co.uk
  data/matches_combined.csv    - Merge de ambas fuentes (por temporada + local + visitante)

Tiempo estimado: ~10-15 minutos (por rate limits de FBref).

Para regenerar solo matches_combined.csv sin volver a descargar:
  python scrape_matches.py --solo-combinar
"""

import os
import re
import pandas as pd
import requests
from scraper_config import SEASONS_FBREF, SEASONS_FOOTBALLDATA, OUTPUT_DIR


def scrape_fbref_matches() -> pd.DataFrame:
    """Extrae fixtures y resultados de FBref usando soccerdata."""
    import soccerdata as sd

    all_dfs = []
    for season in SEASONS_FBREF:
        season_code = season[:4] + season[5:]  # "2023-2024" -> "20232024"
        season_sd = season_code[:4] + "-" + season_code[4:]  # "2023-2024"
        print(f"  [FBref] Descargando {season}...")

        try:
            fbref = sd.FBref(leagues="ENG-Premier League", seasons=season_sd)
            schedule = fbref.read_schedule()

            # Reset index para tener columnas planas
            df = schedule.reset_index()

            # Parsear score
            if "score" in df.columns:
                scores = df["score"].astype(str).str.extract(r"(\d+)\s*[–\-]\s*(\d+)")
                df["home_goals"] = pd.to_numeric(scores[0], errors="coerce")
                df["away_goals"] = pd.to_numeric(scores[1], errors="coerce")

            df["season"] = season
            all_dfs.append(df)
            print(f"    -> {len(df)} partidos")

        except Exception as e:
            print(f"    [ERROR] {season}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def scrape_footballdata_matches() -> pd.DataFrame:
    """
    Descarga CSVs de football-data.co.uk con resultados y odds de apuestas.
    No tiene protección Cloudflare.
    """
    base_url = "https://www.football-data.co.uk/mmz4281"
    all_dfs = []

    for season, code in SEASONS_FOOTBALLDATA.items():
        url = f"{base_url}/{code}/E0.csv"
        print(f"  [football-data] Descargando {season} ({url})...")

        try:
            df = pd.read_csv(url, encoding="utf-8", on_bad_lines="skip")
            df["season"] = season
            all_dfs.append(df)
            print(f"    -> {len(df)} partidos, {len(df.columns)} columnas")
        except Exception as e:
            print(f"    [ERROR] {season}: {e}")
            # Intentar con encoding alternativo
            try:
                df = pd.read_csv(url, encoding="latin-1", on_bad_lines="skip")
                df["season"] = season
                all_dfs.append(df)
                print(f"    -> {len(df)} partidos (latin-1)")
            except Exception as e2:
                print(f"    [ERROR] Segundo intento: {e2}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def combine_sources(fbref_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina los datos de FBref (xG) con los de football-data.co.uk (odds).
    Hace merge por fecha y equipos.
    """
    if fbref_df.empty or odds_df.empty:
        print("  [WARN] Una de las fuentes está vacía, retornando lo disponible")
        return fbref_df if not fbref_df.empty else odds_df

    # Normalizar columnas de FBref
    fb = fbref_df.copy()
    fb_cols = {
        "date": "date",
        "home_team": "home",
        "away_team": "away",
        "home_goals": "home_goals",
        "away_goals": "away_goals",
        "season": "season",
        "week": "gameweek",
        "time": "time",
        "attendance": "attendance",
        "referee": "referee",
        "venue": "venue",
    }
    fb = fb.rename(columns={k: v for k, v in fb_cols.items() if k in fb.columns})
    fb["date"] = pd.to_datetime(fb["date"], errors="coerce")

    # Normalizar columnas de football-data
    od = odds_df.copy()
    od_cols = {
        "Date": "date",
        "HomeTeam": "home_fd",
        "AwayTeam": "away_fd",
        "FTHG": "fd_home_goals",
        "FTAG": "fd_away_goals",
        "FTR": "fd_result",
        "HTHG": "ht_home_goals",
        "HTAG": "ht_away_goals",
        "HTR": "ht_result",
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
        "B365H": "odds_home_b365",
        "B365D": "odds_draw_b365",
        "B365A": "odds_away_b365",
        "BWH": "odds_home_bw",
        "BWD": "odds_draw_bw",
        "BWA": "odds_away_bw",
        "season": "season",
    }
    od = od.rename(columns={k: v for k, v in od_cols.items() if k in od.columns})
    od["date"] = pd.to_datetime(od["date"], format="mixed", dayfirst=True, errors="coerce")

    # Seleccionar solo columnas relevantes de odds
    odds_cols = [c for c in od.columns if c in od_cols.values()]
    od = od[odds_cols]

    # Crear resultado en FBref data
    fb["result"] = "D"
    fb.loc[fb["home_goals"] > fb["away_goals"], "result"] = "H"
    fb.loc[fb["home_goals"] < fb["away_goals"], "result"] = "A"

    # Guardar ambos por separado primero
    save_df(fb, "matches_fbref.csv")
    save_df(od, "matches_odds.csv")

    return merge_matches(fb, od)


# Nombres de equipo de football-data.co.uk -> nombre usado por FBref.
# Los 16 equipos restantes se llaman igual en ambas fuentes.
TEAM_NAME_MAP = {
    "Cardiff": "Cardiff City",
    "Huddersfield": "Huddersfield Town",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Luton": "Luton Town",
    "Man City": "Manchester City",
    "Man United": "Manchester Utd",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
    "Tottenham": "Tottenham Hotspur",
    "West Ham": "West Ham United",
}


def merge_matches(fb: pd.DataFrame, od: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza FBref con football-data.co.uk por (temporada, local, visitante).

    Cada pareja local-visitante se juega una sola vez por temporada, por lo que
    esa llave identifica el partido de forma única. La versión anterior cruzaba
    por (fecha, goles, temporada) sin equipos: cuando dos partidos del mismo día
    terminaban con el mismo marcador el cruce se multiplicaba y asignaba cuotas
    de otro partido (3.730 filas en vez de 3.040).
    """
    fb = fb.copy()
    od = od.copy()
    od["home_key"] = od["home_fd"].replace(TEAM_NAME_MAP)
    od["away_key"] = od["away_fd"].replace(TEAM_NAME_MAP)
    fb["merge_key"] = fb["season"] + "|" + fb["home"] + "|" + fb["away"]
    od["merge_key"] = od["season"] + "|" + od["home_key"] + "|" + od["away_key"]

    # La llave debe ser única en ambas fuentes y todos los partidos deben cruzar
    for nombre, df in (("FBref", fb), ("football-data", od)):
        dup = df["merge_key"].duplicated().sum()
        if dup:
            raise ValueError(f"{nombre}: {dup} partidos repetidos por (temporada, local, visitante)")

    combined = fb.merge(od, on="merge_key", how="left", suffixes=("", "_od"), validate="one_to_one", indicator=True)
    sin_pareja = int((combined["_merge"] != "both").sum())
    if sin_pareja:
        faltan = combined.loc[combined["_merge"] != "both", "merge_key"].tolist()[:10]
        raise ValueError(f"{sin_pareja} partidos de FBref sin cuotas. Revisar TEAM_NAME_MAP. Ejemplos: {faltan}")

    # Los goles de ambas fuentes deben coincidir (control de calidad del cruce)
    distintos = int(
        ((combined["home_goals"] != combined["fd_home_goals"]) | (combined["away_goals"] != combined["fd_away_goals"])).sum()
    )
    if distintos:
        raise ValueError(f"{distintos} partidos con marcador distinto entre FBref y football-data")

    # Limpiar columnas auxiliares y duplicadas
    drop_cols = [c for c in combined.columns if c.endswith("_od")]
    combined = combined.drop(columns=drop_cols + ["merge_key", "home_key", "away_key", "_merge"], errors="ignore")
    return combined


def save_df(df: pd.DataFrame, filename: str):
    """Guarda DataFrame como CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [SAVED] {path} ({len(df)} filas)")


def combine_only():
    """Regenera matches_combined.csv a partir de matches_fbref.csv y matches_odds.csv ya descargados."""
    fb = pd.read_csv(os.path.join(OUTPUT_DIR, "matches_fbref.csv"))
    od = pd.read_csv(os.path.join(OUTPUT_DIR, "matches_odds.csv"))
    combined = merge_matches(fb, od)
    save_df(combined, "matches_combined.csv")
    print(f"TOTAL: {len(combined)} partidos, {len(combined.columns)} columnas")


def main():
    import sys

    if "--solo-combinar" in sys.argv:
        print("Modo --solo-combinar: se reutilizan matches_fbref.csv y matches_odds.csv (sin descargar).")
        combine_only()
        return

    print("=" * 60)
    print("SCRAPING RESULTADOS DE PARTIDOS - PREMIER LEAGUE")
    print("=" * 60)

    # Fuente 1: FBref (fixtures + xG)
    print("\n--- FUENTE 1: FBref ---")
    fbref_df = scrape_fbref_matches()

    # Fuente 2: football-data.co.uk (resultados + odds + stats in-game)
    print("\n--- FUENTE 2: football-data.co.uk ---")
    odds_df = scrape_footballdata_matches()

    # Combinar
    print("\n--- COMBINANDO FUENTES ---")
    combined = combine_sources(fbref_df, odds_df)

    if not combined.empty:
        save_df(combined, "matches_combined.csv")
        print(f"\n{'=' * 60}")
        print(f"TOTAL: {len(combined)} partidos")
        print(f"Columnas: {len(combined.columns)}")
        print(f"\nDistribución de resultados:")
        if "result" in combined.columns:
            print(combined["result"].value_counts().to_string())


if __name__ == "__main__":
    main()
