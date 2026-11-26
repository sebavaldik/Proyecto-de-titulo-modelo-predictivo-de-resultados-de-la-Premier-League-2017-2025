"""
preparar_2025_26.py — Prepara la temporada 2025/26 como prueba real fuera de muestra
=====================================================================================

La temporada 2025/26 no se usó en ninguna decisión del proyecto (ni en la selección de variables ni
en el ajuste de modelos), por lo que sirve para probar los modelos contra la realidad.

Pasos:
  1. Descarga los resultados, estadísticas y cuotas de football-data.co.uk (E0, temporada 2526).
  2. Los convierte al formato de matches_combined.csv (nombres de equipos de FBref) y los agrega
     a los 3.040 partidos validados.
  3. Ejecuta build_features.py y pipeline_features.py sobre esa carpeta (sin tocar Dataset/).
  4. Verifica la calidad: que los partidos originales no cambien y que la forma reciente de los
     partidos nuevos coincida con un recálculo independiente que usa solo partidos anteriores.

Salida: Dataset/prueba_2025_26/ (model_dataset_enriched.csv con 3.420 partidos).

Uso (desde la raíz del proyecto):
    python soccerdata/preparar_2025_26.py
"""

import io
import os
import subprocess
import sys
import urllib.request

import numpy as np
import pandas as pd

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SALIDA = os.path.join(RAIZ, "Dataset", "prueba_2025_26")
URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
TEMPORADA = "2025-2026"

sys.path.insert(0, os.path.join(RAIZ, "soccerdata"))
sys.path.insert(0, RAIZ)
from scrape_matches import TEAM_NAME_MAP  # noqa: E402  (football-data -> FBref)
from utils.validacion_datos import _referencia_rolling  # noqa: E402

COLS_FD = {
    "HS": "home_shots", "AS": "away_shots", "HST": "home_shots_target", "AST": "away_shots_target",
    "HF": "home_fouls", "AF": "away_fouls", "HC": "home_corners", "AC": "away_corners",
    "HY": "home_yellow", "AY": "away_yellow", "HR": "home_red", "AR": "away_red",
    "B365H": "odds_home_b365", "B365D": "odds_draw_b365", "B365A": "odds_away_b365",
}


def descargar():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    crudo = urllib.request.urlopen(req, timeout=60).read()
    with open(os.path.join(SALIDA, "E0_2526_raw.csv"), "wb") as f:
        f.write(crudo)
    return pd.read_csv(io.BytesIO(crudo), encoding="latin-1", on_bad_lines="skip")


def convertir(fd):
    fd = fd.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    fecha = pd.to_datetime(fd["Date"], format="%d/%m/%Y", errors="coerce")
    fecha = fecha.fillna(pd.to_datetime(fd["Date"], format="%d/%m/%y", errors="coerce"))
    out = pd.DataFrame({
        "season": TEMPORADA,
        "date": fecha.dt.strftime("%Y-%m-%d"),
        "home": fd["HomeTeam"].replace(TEAM_NAME_MAP),
        "away": fd["AwayTeam"].replace(TEAM_NAME_MAP),
        "home_goals": fd["FTHG"].astype(int),
        "away_goals": fd["FTAG"].astype(int),
        "result": fd["FTR"],
    })
    for origen, destino in COLS_FD.items():
        out[destino] = fd[origen].values
    return out


def ejecutar(cmd, cwd, env_extra):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", **env_extra)
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise SystemExit(f"Falló: {' '.join(cmd)}")
    return r.stdout


def main():
    os.makedirs(SALIDA, exist_ok=True)
    print("[1] Descargando 2025/26 de football-data.co.uk ...")
    nuevo = convertir(descargar())
    print(f"    {len(nuevo)} partidos, {nuevo['date'].min()} a {nuevo['date'].max()}")

    orig = pd.read_csv(os.path.join(RAIZ, "Dataset", "matches_combined.csv"))
    equipos_previos = set(orig["home"]) | set(orig["away"])
    nuevos_equipos = sorted((set(nuevo["home"]) | set(nuevo["away"])) - equipos_previos)
    print(f"    Equipos sin historial en 2017-2025 (ascendidos): {nuevos_equipos}")

    # controles de los partidos nuevos
    assert len(nuevo) == 380, f"se esperaban 380 partidos y hay {len(nuevo)}"
    assert not nuevo.duplicated(["home", "away"]).any(), "partidos repetidos"
    assert nuevo[["odds_home_b365", "odds_draw_b365", "odds_away_b365"]].notna().all().all(), "faltan cuotas"
    assert (nuevo["date"] >= "2025-08-01").all() and (nuevo["date"] <= "2026-06-30").all(), "fechas fuera de temporada"
    esperado = np.where(nuevo["home_goals"] > nuevo["away_goals"], "H", np.where(nuevo["home_goals"] < nuevo["away_goals"], "A", "D"))
    assert (nuevo["result"].values == esperado).all(), "'result' incoherente con los goles"

    ext = pd.concat([orig, nuevo], ignore_index=True)
    ext.to_csv(os.path.join(SALIDA, "matches_combined.csv"), index=False)
    print(f"[2] matches_combined extendido: {len(ext)} partidos")

    print("[3] build_features.py y pipeline_features.py sobre la carpeta de la prueba ...")
    ejecutar([sys.executable, "build_features.py"], os.path.join(RAIZ, "soccerdata"), {"SOCCERDATA_DIR": SALIDA})
    ejecutar([sys.executable, "pipeline_features.py"], RAIZ, {"DATASET_DIR": SALIDA, "PLAYER_STATS_DIR": os.path.join(RAIZ, "Dataset"), "IMPUTE_UNTIL_SEASON": "2024-2025"})

    print("[4] Verificaciones de calidad ...")
    md = pd.read_csv(os.path.join(SALIDA, "model_dataset.csv"))
    en = pd.read_csv(os.path.join(SALIDA, "model_dataset_enriched.csv"))
    en0 = pd.read_csv(os.path.join(RAIZ, "Dataset", "model_dataset_enriched.csv"))
    md0 = pd.read_csv(os.path.join(RAIZ, "Dataset", "model_dataset.csv"))
    assert not en.duplicated(["season", "home", "away"]).any() and en.isna().sum().sum() == 0
    # (a) los partidos originales no cambian al agregar la temporada nueva (las variables solo dependen del pasado)
    clave = ["season", "home", "away"]
    m = md0.merge(md, on=clave, suffixes=("_0", "_e"))
    num = [c[:-2] for c in m.columns if c.endswith("_0") and c[:-2] + "_e" in m.columns and pd.api.types.is_numeric_dtype(m[c])]
    maxdif = max(float((m[c + "_0"] - m[c + "_e"]).abs().max()) for c in num)
    print(f"    (a) model_dataset: {len(m)} partidos originales; diferencia máxima al extender = {maxdif:.2e}")
    assert maxdif < 1e-9, "las variables de partidos antiguos cambiaron al agregar 2025/26"
    # (a2) los partidos de entrenamiento del enriched son idénticos a los datos ya validados (se compara por partido,
    #      no por posición: al ordenar por fecha los partidos del mismo día pueden quedar en otro orden)
    e_old = en[en["season"] != TEMPORADA]
    mm = en0.merge(e_old, on=clave, suffixes=("_0", "_e"))
    num_e = [c for c in en0.columns if c not in clave and pd.api.types.is_numeric_dtype(en0[c])]
    dif = max(float((mm[c + "_0"] - mm[c + "_e"]).abs().max()) for c in num_e)
    print(f"    (a2) enriched: {len(mm)} partidos de 2017-2025 idénticos a los validados; diferencia máxima = {dif:.2e}")
    assert len(mm) == len(en0) == len(e_old) and dif < 1e-9, "los datos de entrenamiento cambiaron al agregar 2025/26"
    # (b) forma reciente de TODOS los partidos == recálculo independiente con partidos anteriores
    ext["date"] = pd.to_datetime(ext["date"], format="ISO8601")
    ref = _referencia_rolling(ext).set_index(["team", "date"])
    md["date"] = pd.to_datetime(md["date"], format="ISO8601")
    ok = True
    for lado in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([md[lado], md["date"]])
        ok &= np.allclose(md[f"{lado}_roll_goals_scored"].values, ref["roll_gf"].reindex(idx).values, equal_nan=True, atol=1e-9)
        ok &= np.allclose(md[f"{lado}_roll_points"].values, ref["roll_pts"].reindex(idx).values, equal_nan=True, atol=1e-9)
        ok &= np.allclose(md[f"{lado}_unbeaten_streak"].values, ref["racha"].reindex(idx).values, equal_nan=True)
    print(f"    (b) forma reciente == recálculo independiente (con partidos anteriores): {ok}")
    assert ok
    n_nuevo = int((en["season"] == TEMPORADA).sum())
    print(f"    (c) enriched: {len(en)} partidos ({len(en0)} originales + {n_nuevo} de 2025/26), {en.shape[1]} columnas")
    # (d) los partidos que el pipeline descarta son solo los de equipos sin historial en la liga (ascendidos),
    #     igual que se hace con los primeros partidos de la serie original
    claves = ["home", "away"]
    nuevos_ext = ext[ext["season"] == TEMPORADA][claves]
    presentes = en[en["season"] == TEMPORADA][claves]
    excluidos = nuevos_ext.merge(presentes, on=claves, how="left", indicator=True).query("_merge == 'left_only'")
    print(f"    (d) partidos de 2025/26 excluidos por falta de historial: {len(excluidos)} -> "
          + "; ".join(f"{r.home}-{r.away}" for r in excluidos.itertuples()))
    assert len(en) == len(en0) + n_nuevo
    assert all((r.home in nuevos_equipos) or (r.away in nuevos_equipos) for r in excluidos.itertuples()), \
        "se excluyeron partidos de equipos con historial: revisar"
    print("OK: Dataset/prueba_2025_26/ listo")


if __name__ == "__main__":
    main()
