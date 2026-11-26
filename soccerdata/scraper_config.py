"""
Configuración del scraper de datos de la Premier League.
"""

# Temporadas a scrapear (desde 2017-2018 que tiene datos avanzados como xG en FBref)
SEASONS_FBREF = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]

# Mapeo de temporadas a códigos de football-data.co.uk
SEASONS_FOOTBALLDATA = {
    "2017-2018": "1718",
    "2018-2019": "1819",
    "2019-2020": "1920",
    "2020-2021": "2021",
    "2021-2022": "2122",
    "2022-2023": "2223",
    "2023-2024": "2324",
    "2024-2025": "2425",
}

# Directorio de salida (se puede cambiar con la variable de entorno SOCCERDATA_DIR,
# p. ej. SOCCERDATA_DIR=../Dataset para trabajar directamente sobre la carpeta Dataset/)
import os
OUTPUT_DIR = os.environ.get("SOCCERDATA_DIR", "data")

# Categorías de estadísticas disponibles en soccerdata/FBref
# Válidos: standard, keeper, shooting, playing_time, misc
TEAM_STAT_TYPES = [
    "standard",
    "keeper",
    "shooting",
    "playing_time",
    "misc",
]

PLAYER_STAT_TYPES = [
    "standard",
    "keeper",
    "shooting",
    "playing_time",
    "misc",
]
