"""
validacion_datos.py — Controles de calidad del dataset
======================================================

Verifica que los datos no tengan los dos errores detectados en el proyecto:

1. Partidos duplicados / cuotas de otro partido (cruce FBref + football-data sin equipos).
2. Fechas con día y mes invertidos (dayfirst=True aplicado a fechas ISO), que desordenaba
   el calendario y contaminaba las variables de forma reciente.

Además recalcula de forma INDEPENDIENTE (con otro código que build_features.py) la forma
reciente y las rachas, para confirmar que cada partido solo usa partidos anteriores.

Uso (desde la raíz del proyecto):
    python utils/validacion_datos.py

Genera resultados/validacion_dataset.txt y termina con código 1 si algún control falla.
"""

import os
import sys

import numpy as np
import pandas as pd

# Nombres de football-data.co.uk -> FBref (mismo mapa que soccerdata/scrape_matches.py)
MAPA_EQUIPOS = {
    "Cardiff": "Cardiff City", "Huddersfield": "Huddersfield Town", "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United", "Leicester": "Leicester City", "Luton": "Luton Town",
    "Man City": "Manchester City", "Man United": "Manchester Utd", "Newcastle": "Newcastle United",
    "Norwich": "Norwich City", "Nott'm Forest": "Nottingham Forest", "Stoke": "Stoke City",
    "Swansea": "Swansea City", "Tottenham": "Tottenham Hotspur", "West Ham": "West Ham United",
}
VENTANA = 5  # ventana de forma reciente usada en build_features.py


class Reporte:
    def __init__(self):
        self.lineas, self.fallos = [], 0

    def check(self, nombre, ok, detalle=""):
        self.lineas.append(f"[{'OK ' if ok else 'FALLA'}] {nombre}" + (f" — {detalle}" if detalle else ""))
        self.fallos += 0 if ok else 1

    def titulo(self, texto):
        self.lineas += ["", texto, "-" * len(texto)]


def _eventos(partidos: pd.DataFrame) -> pd.DataFrame:
    """Una fila por equipo y partido, con goles a favor/en contra y puntos."""
    pts_local = np.where(partidos["home_goals"] > partidos["away_goals"], 3, np.where(partidos["home_goals"] == partidos["away_goals"], 1, 0))
    pts_visita = np.where(partidos["away_goals"] > partidos["home_goals"], 3, np.where(partidos["home_goals"] == partidos["away_goals"], 1, 0))
    local = pd.DataFrame({"date": partidos["date"], "team": partidos["home"], "gf": partidos["home_goals"], "gc": partidos["away_goals"], "pts": pts_local})
    visita = pd.DataFrame({"date": partidos["date"], "team": partidos["away"], "gf": partidos["away_goals"], "gc": partidos["home_goals"], "pts": pts_visita})
    return pd.concat([local, visita]).sort_values(["team", "date"]).reset_index(drop=True)


def _referencia_rolling(partidos: pd.DataFrame) -> pd.DataFrame:
    """Forma reciente y racha invicta calculadas solo con partidos ANTERIORES, por equipo."""
    ev = _eventos(partidos)
    g = ev.groupby("team")
    ev["roll_gf"] = g["gf"].transform(lambda s: s.rolling(VENTANA, min_periods=1).mean().shift(1))
    ev["roll_pts"] = g["pts"].transform(lambda s: s.rolling(VENTANA, min_periods=1).mean().shift(1))

    def racha(pts):
        out, actual = [], 0
        for p in pts:
            out.append(actual)
            actual = actual + 1 if p > 0 else 0
        return pd.Series(out, index=pts.index)

    ev["racha"] = g["pts"].transform(racha)
    return ev


def validar(raiz: str = ".") -> Reporte:
    R = Reporte()
    ds = lambda f: os.path.join(raiz, "Dataset", f)

    # ------------------------------------------------------------------ matches_combined
    R.titulo("1. matches_combined.csv (cruce FBref + football-data)")
    fb = pd.read_csv(ds("matches_fbref.csv"))
    cb = pd.read_csv(ds("matches_combined.csv"))
    cb["date"] = pd.to_datetime(cb["date"], format="ISO8601")
    R.check("3.040 partidos (8 temporadas x 380)", len(cb) == 3040, f"{len(cb)} filas")
    R.check("380 partidos por temporada", (cb.groupby("season").size() == 380).all(), str(cb.groupby("season").size().to_dict()))
    R.check("Sin partidos repetidos por (temporada, local, visitante)", not cb.duplicated(["season", "home", "away"]).any())
    ev = _eventos(cb)
    R.check("Ningún equipo juega dos partidos el mismo día", not ev.duplicated(["team", "date"]).any())
    R.check("Las cuotas pertenecen al partido correcto (equipos coinciden)",
            ((cb["home_fd"].replace(MAPA_EQUIPOS) == cb["home"]) & (cb["away_fd"].replace(MAPA_EQUIPOS) == cb["away"])).all())
    R.check("Marcador FBref == marcador football-data", ((cb["home_goals"] == cb["fd_home_goals"]) & (cb["away_goals"] == cb["fd_away_goals"])).all())
    esperado = np.where(cb["home_goals"] > cb["away_goals"], "H", np.where(cb["home_goals"] < cb["away_goals"], "A", "D"))
    R.check("'result' coherente con los goles", (cb["result"].values == esperado).all())
    R.check("Cuotas B365 completas y > 1", cb[["odds_home_b365", "odds_draw_b365", "odds_away_b365"]].notna().all().all()
            and (cb[["odds_home_b365", "odds_draw_b365", "odds_away_b365"]] > 1).all().all())
    fb_fecha = pd.to_datetime(fb.set_index(["season", "home", "away"])["date"], format="ISO8601")
    cb_fecha = cb.set_index(["season", "home", "away"])["date"]
    R.check("Fechas == fechas de matches_fbref.csv",
            cb_fecha.index.is_unique and bool((cb_fecha.reindex(fb_fecha.index) == fb_fecha).all()))
    inicio = cb["season"].str[:4].astype(int); fin = cb["season"].str[5:].astype(int)
    dentro = (cb["date"] >= pd.to_datetime(inicio.astype(str) + "-08-01")) & (cb["date"] <= pd.to_datetime(fin.astype(str) + "-08-31"))
    R.check("Cada fecha cae dentro de su temporada (ago año1 - ago año2)", dentro.all(), f"{int((~dentro).sum())} fuera de rango")

    # ------------------------------------------------------------------ model_dataset
    R.titulo("2. model_dataset.csv (variables rolling)")
    md = pd.read_csv(ds("model_dataset.csv"))
    md["date"] = pd.to_datetime(md["date"], format="ISO8601")
    R.check("Sin partidos repetidos", not md.duplicated(["season", "home", "away"]).any(), f"{len(md)} filas")
    clave = md.merge(cb[["season", "home", "away", "date"]].rename(columns={"date": "fecha_real"}), on=["season", "home", "away"], how="left")
    R.check("Fecha == fecha real del partido (sin día/mes invertidos)", (clave["date"] == clave["fecha_real"]).all(), f"{int((clave['date'] != clave['fecha_real']).sum())} distintas")
    R.check("month == mes de la fecha", (md["month"] == md["date"].dt.month).all())
    R.check("day_of_week == día de la semana", (md["day_of_week"] == md["date"].dt.dayofweek).all())
    R.check("is_weekend coherente", (md["is_weekend"] == (md["date"].dt.dayofweek >= 5).astype(int)).all())
    # cuotas del partido correcto
    cuotas = md.merge(cb[["season", "home", "away", "odds_home_b365", "odds_draw_b365", "odds_away_b365"]], on=["season", "home", "away"], how="left")
    R.check("odds_home/draw/away == cuotas B365 de ese partido",
            np.allclose(cuotas["odds_home"], cuotas["odds_home_b365"]) and np.allclose(cuotas["odds_draw"], cuotas["odds_draw_b365"]) and np.allclose(cuotas["odds_away"], cuotas["odds_away_b365"]))
    # forma reciente recalculada de forma independiente
    ref = _referencia_rolling(cb).set_index(["team", "date"])
    if not ref.index.is_unique:
        R.check("Forma reciente == recálculo independiente", False, "no se puede recalcular: hay partidos duplicados por equipo y fecha")
    else:
        for lado, col_equipo in (("home", "home"), ("away", "away")):
            idx = pd.MultiIndex.from_arrays([md[col_equipo], md["date"]])
            ok_gf = np.allclose(md[f"{lado}_roll_goals_scored"].values, ref["roll_gf"].reindex(idx).values, equal_nan=True, atol=1e-9)
            ok_pt = np.allclose(md[f"{lado}_roll_points"].values, ref["roll_pts"].reindex(idx).values, equal_nan=True, atol=1e-9)
            ok_rc = np.allclose(md[f"{lado}_unbeaten_streak"].values, ref["racha"].reindex(idx).values, equal_nan=True)
            R.check(f"{lado}: roll_goals_scored, roll_points y unbeaten_streak == recálculo independiente solo con partidos anteriores", ok_gf and ok_pt and ok_rc)

    # ------------------------------------------------------------------ enriched
    R.titulo("3. model_dataset_enriched.csv (entrada de los modelos)")
    en = pd.read_csv(ds("model_dataset_enriched.csv"))
    en["date"] = pd.to_datetime(en["date"], format="ISO8601")
    R.check("Sin partidos repetidos", not en.duplicated(["season", "home", "away"]).any(), f"{len(en)} filas x {en.shape[1]} columnas")
    R.check("Sin valores nulos", int(en.isna().sum().sum()) == 0)
    tgt = en.merge(md[["season", "home", "away", "result"]], on=["season", "home", "away"], how="left")
    R.check("target_home_win == (result == 'H')", (tgt["target_home_win"] == (tgt["result"] == "H").astype(int)).all())
    R.lineas.append("Filas por temporada: " + ", ".join(f"{s_}={n}" for s_, n in en.groupby("season").size().items()))
    R.lineas.append("")
    R.lineas.append(f"Victorias locales: {en['target_home_win'].mean() * 100:.2f} %  |  filas: {len(en)}")
    R.lineas.append("RESULTADO GLOBAL: " + ("TODOS LOS CONTROLES OK" if R.fallos == 0 else f"{R.fallos} CONTROL(ES) FALLARON"))
    return R


if __name__ == "__main__":
    raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reporte = validar(raiz)
    texto = "\n".join(reporte.lineas)
    print(texto)
    os.makedirs(os.path.join(raiz, "resultados"), exist_ok=True)
    with open(os.path.join(raiz, "resultados", "validacion_dataset.txt"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    sys.exit(1 if reporte.fallos else 0)
