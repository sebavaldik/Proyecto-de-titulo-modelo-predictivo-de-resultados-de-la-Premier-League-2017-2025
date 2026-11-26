"""
Script 3: Extrae estadísticas de jugadores por temporada desde FBref.

Genera un CSV por categoría en data/:
  player_stats_standard.csv
  player_stats_keeper.csv
  player_stats_shooting.csv
  player_stats_playing_time.csv
  player_stats_misc.csv

Tiempo estimado: ~15-20 minutos.
"""

import os
import pandas as pd
import soccerdata as sd
from scraper_config import SEASONS_FBREF, PLAYER_STAT_TYPES, OUTPUT_DIR


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana MultiIndex columns a strings simples."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            f"{a}_{b}".strip("_") if b else a
            for a, b in df.columns
        ]
    return df


def scrape_player_stats():
    """Extrae estadísticas de jugadores para todas las temporadas y categorías."""
    accumulators = {stat: [] for stat in PLAYER_STAT_TYPES}

    for season in SEASONS_FBREF:
        print(f"\n[{season}]")

        try:
            fbref = sd.FBref(leagues="ENG-Premier League", seasons=season)
        except Exception as e:
            print(f"  [ERROR] Inicializando FBref para {season}: {e}")
            continue

        for stat_type in PLAYER_STAT_TYPES:
            print(f"  Categoría: {stat_type}...", end=" ")

            try:
                df = fbref.read_player_season_stats(stat_type=stat_type)
                df = df.reset_index()
                df = flatten_columns(df)

                # Renombrar "season" de FBref para no colisionar
                if "season" in df.columns:
                    df.rename(columns={"season": "season_code"}, inplace=True)
                df.insert(0, "season_label", season)

                accumulators[stat_type].append(df)
                print(f"-> {len(df)} jugadores")

            except Exception as e:
                print(f"ERROR: {e}")

    return accumulators


def main():
    print("=" * 60)
    print("SCRAPING ESTADÍSTICAS DE JUGADORES - PREMIER LEAGUE")
    print("=" * 60)

    accumulators = scrape_player_stats()

    print(f"\n{'=' * 60}")
    print("GUARDANDO ARCHIVOS...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for stat_type, dfs in accumulators.items():
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            path = os.path.join(OUTPUT_DIR, f"player_stats_{stat_type}.csv")
            combined.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  [SAVED] {path} ({len(combined)} filas, {len(combined.columns)} cols)")
        else:
            print(f"  [WARN] Sin datos para '{stat_type}'")

    print("\nDone!")


if __name__ == "__main__":
    main()
