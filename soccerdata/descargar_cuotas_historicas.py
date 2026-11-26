"""
descargar_cuotas_historicas.py — Descarga las cuotas de varias casas de apuestas (football-data.co.uk)
=======================================================================================================

Uso exclusivo del experimento exploratorio `13_experimento_cuotas_mejoradas.ipynb`: guarda el CSV completo de la
Premier League (E0) de cada temporada 2017/18 a 2024/25 en `Dataset/experimento_cuotas/raw/`
(la temporada 2025/26 ya la deja `preparar_2025_26.py` en `Dataset/prueba_2025_26/E0_2526_raw.csv`).

Uso (desde la raíz del proyecto):
    python soccerdata/descargar_cuotas_historicas.py
"""

import os
import shutil
import urllib.request

import pandas as pd

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DESTINO = os.path.join(RAIZ, "Dataset", "experimento_cuotas", "raw")
os.makedirs(DESTINO, exist_ok=True)

for a in range(17, 25):
    codigo = f"{a}{a + 1}"
    salida = os.path.join(DESTINO, f"E0_{codigo}.csv")
    if not os.path.exists(salida):
        req = urllib.request.Request(f"https://www.football-data.co.uk/mmz4281/{codigo}/E0.csv", headers={"User-Agent": "Mozilla/5.0"})
        with open(salida, "wb") as f:
            f.write(urllib.request.urlopen(req, timeout=60).read())
    df = pd.read_csv(salida, encoding="latin-1", on_bad_lines="skip").dropna(subset=["HomeTeam"])
    casas = sorted({c[:-1] for c in df.columns if c.endswith("H") and c[:-1] + "D" in df.columns and c[:-1] + "A" in df.columns
                    and df[c].notna().mean() > 0.9})
    print(codigo, len(df), "partidos; casas con >90 % de cobertura:", casas)

origen_2526 = os.path.join(RAIZ, "Dataset", "prueba_2025_26", "E0_2526_raw.csv")
if os.path.exists(origen_2526):
    shutil.copy(origen_2526, os.path.join(DESTINO, "E0_2526.csv"))
else:
    print("Falta Dataset/prueba_2025_26/E0_2526_raw.csv: ejecutar antes soccerdata/preparar_2025_26.py")
