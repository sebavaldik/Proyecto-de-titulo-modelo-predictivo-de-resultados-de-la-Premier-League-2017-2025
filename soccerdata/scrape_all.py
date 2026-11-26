"""
Script maestro: ejecuta los 3 scrapers en orden.

Uso:
    python scrape_all.py

Tiempo estimado: ~30-45 minutos (por el rate limit de FBref).
Los datos se guardan en la carpeta data/.

Después de ejecutar este script, corre:
    python build_features.py
para generar el dataset listo para modelado.
"""

import time


def main():
    start = time.time()

    print("+" + "=" * 58 + "+")
    print("|  PREMIER LEAGUE DATA SCRAPER                             |")
    print("|  Fuentes: FBref + football-data.co.uk                    |")
    print("+" + "=" * 58 + "+")

    # 1. Partidos (FBref + football-data.co.uk)
    print("\n" + "-" * 60)
    print("FASE 1/3: RESULTADOS DE PARTIDOS")
    print("-" * 60)
    import scrape_matches
    scrape_matches.main()

    # 2. Stats de equipos (FBref)
    print("\n" + "-" * 60)
    print("FASE 2/3: ESTADISTICAS DE EQUIPOS")
    print("-" * 60)
    import scrape_team_stats
    scrape_team_stats.main()

    # 3. Stats de jugadores (FBref)
    print("\n" + "-" * 60)
    print("FASE 3/3: ESTADISTICAS DE JUGADORES")
    print("-" * 60)
    import scrape_player_stats
    scrape_player_stats.main()

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 60)
    print(f"COMPLETADO en {minutes}m {seconds}s")
    print(f"\nArchivos generados en data/:")
    print(f"  Partidos:")
    print(f"    - matches_fbref.csv        (fixtures + xG)")
    print(f"    - matches_odds.csv         (resultados + odds de apuestas)")
    print(f"    - matches_combined.csv     (merge de ambas fuentes)")
    print(f"  Equipos:")
    print(f"    - team_stats_*.csv         (5 archivos: standard, keeper, shooting, playing_time, misc)")
    print(f"  Jugadores:")
    print(f"    - player_stats_*.csv       (5 archivos: standard, keeper, shooting, playing_time, misc)")
    print(f"\nSiguiente paso: python build_features.py")


if __name__ == "__main__":
    main()
