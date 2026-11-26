# Modelo Predictivo de Resultados de Partidos de la Premier League (2017/2025)

Proyecto de Título — Ingeniería Civil Informática, Universidad Andrés Bello.

- **Autores:** Anakin Benavides · Sebastián Valdovinos
- **Profesor Guía:** Pablo Hernán Schwarzenberg Riveros

## Resumen

Se comparan modelos de aprendizaje de máquinas que predicen **una sola cosa: si el equipo local gana un partido de la
Premier League** (`target_home_win` = 1) **o no** (0; el «no» agrupa empate y victoria visitante), con datos de las
temporadas 2017/18 a 2024/25. La evaluación usa **validación temporal de ventana expansiva (7 folds)**: cada fold entrena
con el pasado y prueba en la temporada siguiente.

Resultados con los datos corregidos (promedio de 7 folds):

| Modelo | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest | 68,2 % | 0,727 |
| Red neuronal PyTorch | 68,0 % | 0,725 |
| Solo las 3 cuotas (Reg. Logística, referencia) | 67,7 % (68,6 % calibrada) | **0,736** |
| Regresión Logística (28 variables) | 67,4 % | 0,731 |
| Red neuronal Keras Tuner | 66,8 % | 0,716 |
| Gradient Boosting | 65,8 % | 0,704 |
| MLP (sklearn) | 65,8 % | 0,703 |
| SVM (RBF) | 61,1 % | 0,690 |
| *Regla simple del mercado (referencia)* | *64,4 %* | — |

**Conclusión:** los modelos superan a la regla simple del mercado en 3 a 4 puntos (Random Forest, en los 7 folds) y
**llegan al nivel de una regresión logística entrenada solo con las probabilidades de las cuotas**: con datos propios
(forma, plantel, enfrentamientos directos) se reconstruye la capacidad predictiva del mercado. Sin las cuotas el mejor
modelo ya iguala la regla simple del mercado (64,9 % vs. 64,4 %), lo que explica que agregarlas solo suba unos 3 puntos.
La mezcla de modelos tampoco mejora a las cuotas solas. La mejora futura principal es **añadir el empate** (3 clases:
local, empate, visitante). Detalle completo en el informe (`Documento Tesis/informe/`).

## Prueba con datos reales (temporada 2025/26) y análisis de error

La temporada 2025/26 no se usó en ninguna decisión del proyecto, por lo que sirve como prueba fuera de muestra
(`soccerdata/preparar_2025_26.py` descarga y procesa los datos en `Dataset/prueba_2025_26/`, sin tocar los datos originales).

| Modelo (377 partidos) | Accuracy (IC 95 %) | ROC-AUC |
|---|---|---|
| Random Forest | 61,3 % (56,3–66,1) | 0,672 |
| Solo las 3 cuotas | 62,9 % (57,9–67,6) | 0,697 |
| *Regla simple del mercado* | *60,5 % (55,5–65,3)* | — |

Ninguna diferencia es estadísticamente significativa (McNemar, p > 0,3): los modelos no superan de forma demostrable al mercado.

**Análisis de error por equipo** (3.003 partidos, Random Forest): mejor acierto con Manchester City (79,1 %), Liverpool (76,7 %) y
Arsenal (71,9 %); peor con Aston Villa (59,9 %), Brighton (60,4 %) y Nottingham Forest (61,7 %). Solo City y Liverpool difieren del
promedio más allá del azar (corrección de Benjamini-Hochberg). El 72 % de los errores del modelo también los comete el mercado y se
concentran en partidos parejos (55,9 % de acierto) y empates. Tablas en `resultados/analisis_error_*.csv`, figuras en `Imagenes/analisis_error/`.

## Control de calidad de los datos

Una primera versión del conjunto de datos arrojaba 74,65 % de accuracy. Un validador independiente
(`utils/validacion_datos.py`, 21 controles) detectó dos errores que se corrigieron:

1. **Partidos duplicados:** el cruce FBref + cuotas no usaba los equipos y generó 3.730 filas en vez de 3.040, con cuotas
   de otro partido (`soccerdata/scrape_matches.py`).
2. **Fechas con día y mes invertidos** (`dayfirst=True` sobre fechas ISO) en el 35 % de los partidos, lo que dejaba partidos
   futuros dentro de la "forma reciente" (`soccerdata/build_features.py`).

Lección aprendida: una métrica alta se verifica antes de creerla. Los resultados originales se conservan en
`resultados/previo_con_duplicados/` y la comparación en `resultados/comparacion_antes_despues.csv`. El reporte del
validador está en `resultados/validacion_dataset.txt`.

## Estructura del repositorio

```
.
├── soccerdata/                  Scrapers y construcción del dataset (FBref + football-data.co.uk)
│                                (preparar_2025_26.py: temporada de prueba; descargar_cuotas_historicas.py: experimento 13)
├── Dataset sin modificar/       Copia de respaldo de los CSV crudos originales (con los errores)
├── Dataset/                     CSV de trabajo, ya corregidos
│   ├── prueba_2025_26/          Temporada 2025/26 procesada (prueba fuera de muestra)
│   └── experimento_cuotas/raw/  Cuotas de varias casas (experimento exploratorio 13)
├── pipeline_features.py         Feature engineering (genera model_dataset_clean / enriched)
├── utils/                       División temporal, evaluación y validación de datos
├── 01_eda.ipynb … 11_analisis_error.ipynb   Notebooks del análisis (ver orden abajo)
├── 12_*.ipynb, 13_*.ipynb       Experimentos exploratorios, fuera del informe
├── resultados/                  CSV de métricas (los modelos entrenados se regeneran y no se versionan)
├── Imagenes/                    Figuras generadas por los notebooks
├── Documento Tesis/
│   ├── informe/                 Informe de título (ICF_2026 .docx), PDF final y fuente LaTeX (fuente_latex/)
│   ├── presentacion/            Presentación (.pptx) y guion para dos presentadores (.docx)
│   ├── Papers/                  Artículos revisados (PDF de terceros; ver .gitignore)
│   ├── documentos_institucionales/   Guía y formatos UNAB
│   └── versiones_anteriores/    Presentación y guion anteriores
├── DICCIONARIO_DE_DATOS.txt     Descripción de cada columna de cada CSV
├── requirements.txt             Dependencias (Python 3.12)
└── README.md
```

En `Documento Tesis/informe/fuente_latex/`, `informe_final.tex` está en formato IEEE Access y `informe_tesis.tex` en el
formato de tesis de la guía UNAB; `main.tex` es el archivo principal si se compila en Overleaf.

## Cómo reproducir

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

**1. Datos** (opcional, ~30-45 min por el límite de velocidad de FBref; los CSV ya están en `Dataset/`):

```bash
cd soccerdata
python scrape_all.py              # partidos, estadísticas de equipos y de jugadores → soccerdata/data/
python build_features.py          # construye model_dataset.csv
```

Los scrapers escriben en `soccerdata/data/` (o en la carpeta indicada en la variable de entorno `SOCCERDATA_DIR`,
p. ej. `SOCCERDATA_DIR=../Dataset`). Para regenerar solo `matches_combined.csv` sin volver a descargar:
`python scrape_matches.py --solo-combinar`.

**2. Feature engineering y validación** (desde la raíz del proyecto):

```bash
python pipeline_features.py       # → Dataset/model_dataset_clean.csv y model_dataset_enriched.csv
python utils/validacion_datos.py  # 21 controles de calidad → resultados/validacion_dataset.txt
```

**3. Notebooks** (ejecutar en orden, desde la raíz):

| # | Notebook | Qué hace |
|---|---|---|
| 01 | `01_eda.ipynb` | Análisis exploratorio, balance de clases y baseline del mercado (`resultados/baseline_mercado.json`) |
| 02 | `02_modelos.ipynb` | Seis modelos sin optimizar (partición 80/20) |
| 03 | `03_feature_selection.ipynb` | Limpieza de fuga/redundancia y RFECV → `Dataset/model_dataset_selected.csv` (28 variables) |
| 04 | `04_tuning.ipynb` | GridSearchCV por fold (7 folds) + red PyTorch; guarda modelos en `resultados/models/` |
| 05 | `05_sin_odds.ipynb` | Evaluación con y sin variables de mercado |
| 06 | `06_features_negativos.ipynb` | Importancia por permutación y prueba de eliminación → `model_dataset_final.csv` |
| 07 | `07_keras_tuner.ipynb` | Red neuronal con Keras Tuner (optimización bayesiana, promedio de 10 redes por fold) |
| 08 | `08_shap.ipynb` | Interpretabilidad con valores SHAP |
| 09 | `09_mezcla_modelos.ipynb` | Calibración de probabilidades y mezcla de modelos |
| 10 | `10_prueba_real_2025_26.ipynb` | Prueba con la temporada 2025/26 (377 partidos, no usada en ninguna decisión); requiere ejecutar antes `python soccerdata/preparar_2025_26.py` |
| 11 | `11_analisis_error.ipynb` | Análisis de error por equipo, resultado y tipo de partido (3.003 partidos: 7 folds + 2025/26) |

**4. Experimentos exploratorios** (no forman parte del informe):

| # | Notebook | Qué hace y resultado |
|---|---|---|
| 12 | `12_ventana_deslizante.ipynb` | Ventana deslizante (entrenar solo con la temporada anterior) vs. expansiva: no mejora de forma relevante (RF −0,6 pp, solo cuotas +0,4 pp; McNemar p > 0,2) |
| 13 | `13_experimento_cuotas_mejoradas.ipynb` | Consenso de varias casas, método de Shin y corrección sobre el mercado: sin mejora; solo la cuota de cierre de Pinnacle mejora (+0,6 pp), porque incorpora información de las horas previas al partido. Requiere `python soccerdata/descargar_cuotas_historicas.py` si faltan los CSV de `Dataset/experimento_cuotas/raw/` |

## Presentación

`Documento Tesis/presentacion/` contiene la presentación (24 diapositivas + 3 de glosario) y el guion para dos
presentadores, con tiempos, conectores entre diapositivas y preguntas probables. El guion también está en las notas del orador.

## Datos

Fuentes: [FBref](https://fbref.com) (vía la librería `soccerdata`) y [football-data.co.uk](https://www.football-data.co.uk).
Revisar los términos de uso de cada fuente antes de redistribuir los datos. El significado de cada columna está en
`DICCIONARIO_DE_DATOS.txt`.
