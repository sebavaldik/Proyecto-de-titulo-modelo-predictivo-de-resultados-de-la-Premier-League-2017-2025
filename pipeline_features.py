"""
pipeline_features.py — Pipeline definitivo de Feature Engineering
=================================================================
Ejecutar PRIMERO antes de los notebooks de EDA y Modelos.

Genera dos archivos en Dataset/:
  - model_dataset_clean.csv    : dataset limpio listo para EDA
  - model_dataset_enriched.csv : dataset con features de jugadores (input de modelos)

Estrategia anti-leakage en features de jugadores:
  Cada partido en temporada S usa estadísticas de jugadores de S-1.
  Los equipos sin temporada previa reciben la mediana de la liga.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 0. CARGAR DATOS FUENTE
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PIPELINE DE FEATURE ENGINEERING — Premier League")
print("=" * 60)

# Carpetas de entrada/salida (por defecto 'Dataset'; la prueba con 2025/26 usa una carpeta propia para no
# modificar los datos ya validados). Las estadísticas de jugadores se leen de PLAYER_STATS_DIR.
D_OUT = os.environ.get('DATASET_DIR', 'Dataset')
D_PLAYERS = os.environ.get('PLAYER_STATS_DIR', 'Dataset')

df_raw   = pd.read_csv(f'{D_OUT}/model_dataset.csv')
ps_std   = pd.read_csv(f'{D_PLAYERS}/player_stats_standard.csv')
ps_gk    = pd.read_csv(f'{D_PLAYERS}/player_stats_keeper.csv')
ps_misc  = pd.read_csv(f'{D_PLAYERS}/player_stats_misc.csv')
ps_pt    = pd.read_csv(f'{D_PLAYERS}/player_stats_playing_time.csv')

print(f"\n[RAW] model_dataset: {df_raw.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LIMPIEZA BASE Y TARGET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Limpieza base y definición del target...")

# Controles de calidad de la entrada: sin partidos repetidos y con fechas dentro de las temporadas
dup = int(df_raw.duplicated(['season', 'home', 'away']).sum())
assert dup == 0, f"model_dataset.csv tiene {dup} partidos repetidos; regenerar con soccerdata/build_features.py"
_fechas = pd.to_datetime(df_raw['date'], format='ISO8601')
assert _fechas.min() >= pd.Timestamp('2017-07-01') and _fechas.max() <= pd.Timestamp('2026-07-01'), \
    "Fechas fuera del rango 2017/18-2025/26: revisar el parseo de fechas en build_features.py"
print(f"  Controles de entrada OK: {len(df_raw)} partidos únicos, fechas {_fechas.min().date()} a {_fechas.max().date()}")

df = df_raw.copy()
df['target_home_win'] = (df['result'] == 'H').astype(int)

# Eliminar columnas de leakage e identificadores
LEAKAGE = ['home_goals', 'away_goals', 'result']
df.drop(columns=LEAKAGE, inplace=True)

# Nulos (vienen del rolling en primeros partidos de temporada)
n_antes = len(df)
df.dropna(inplace=True)
print(f"  Filas eliminadas por nulos: {n_antes - len(df)} | Restantes: {len(df)}")

# Guardar dataset limpio (para EDA)
df.to_csv(f'{D_OUT}/model_dataset_clean.csv', index=False)
print(f"  Guardado: model_dataset_clean.csv  →  {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURES DE PLANTEL: PORTERO TITULAR
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Features de portero titular (mayor nº de starts)...")

gk_feats = (
    ps_gk.sort_values('Playing Time_Starts', ascending=False)
    .groupby(['season_label', 'team'])
    .first()
    .reset_index()
)[['season_label', 'team', 'Performance_Save%', 'Performance_CS%', 'Performance_GA90']]
gk_feats.columns = ['season_label', 'team', 'gk_save_pct', 'gk_cs_pct', 'gk_ga90']

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURES DE PLANTEL: CALIDAD OFENSIVA Y PERFIL
# ─────────────────────────────────────────────────────────────────────────────
print("[3] Features ofensivas y perfil de plantel...")

def squad_features(group):
    g = group.sort_values('Playing Time_Min', ascending=False)
    outfield = g[~g['pos'].str.startswith('GK', na=False)].head(3)
    top3_ga  = outfield['Per 90 Minutes_G+A'].sum()
    total_min = g['Playing Time_Min'].sum()
    age_w = (g['age'] * g['Playing Time_Min']).sum() / total_min if total_min > 0 else g['age'].mean()
    depth = (g['Playing Time_Min'] > 900).sum()
    return pd.Series({'top3_ga_per90': top3_ga, 'squad_age': age_w, 'squad_depth': depth})

squad_feats = (
    ps_std.groupby(['season_label', 'team'])
    .apply(squad_features, include_groups=False)
    .reset_index()
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURES DE PLANTEL: TRABAJO DEFENSIVO Y DISCIPLINA
# ─────────────────────────────────────────────────────────────────────────────
print("[4] Features defensivas y disciplina...")

def defensive_features(group):
    t90 = group['90s'].sum()
    if t90 == 0:
        return pd.Series({'team_tkl_per90': np.nan, 'team_int_per90': np.nan, 'team_discipline_per90': np.nan})
    return pd.Series({
        'team_tkl_per90':         group['Performance_TklW'].sum() / t90,
        'team_int_per90':         group['Performance_Int'].sum()  / t90,
        'team_discipline_per90': (group['Performance_CrdY'].sum() + group['Performance_CrdR'].sum() * 3) / t90,
    })

def_feats = (
    ps_misc.groupby(['season_label', 'team'])
    .apply(defensive_features, include_groups=False)
    .reset_index()
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURES DE PLANTEL: IMPACTO (ON-OFF)
# ─────────────────────────────────────────────────────────────────────────────
print("[5] Features de impacto (On-Off)...")

def impact_features(group):
    key = group[group['Playing Time_Min'] > 1000]
    if len(key) == 0:
        key = group
    return pd.Series({'avg_on_off': key['Team Success_On-Off'].mean()})

impact_feats = (
    ps_pt.groupby(['season_label', 'team'])
    .apply(impact_features, include_groups=False)
    .reset_index()
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. MERGE DE FEATURES DE PLANTEL
# ─────────────────────────────────────────────────────────────────────────────
print("[6] Mergeando features de plantel...")

ptf = (
    gk_feats
    .merge(squad_feats, on=['season_label', 'team'], how='left')
    .merge(def_feats,   on=['season_label', 'team'], how='left')
    .merge(impact_feats,on=['season_label', 'team'], how='left')
)
ptf.to_csv(f'{D_OUT}/player_team_features.csv', index=False)

PLAYER_COLS = [
    'gk_save_pct', 'gk_cs_pct', 'gk_ga90',
    'top3_ga_per90', 'squad_age', 'squad_depth',
    'team_tkl_per90', 'team_int_per90', 'team_discipline_per90',
    'avg_on_off'
]

# ─────────────────────────────────────────────────────────────────────────────
# 7. JOIN CON MODEL_DATASET USANDO TEMPORADA ANTERIOR (S-1)
# ─────────────────────────────────────────────────────────────────────────────
print("[7] Join con model_dataset usando temporada S-1...")

seasons_sorted = sorted(ps_std['season_label'].unique())
next_of = {seasons_sorted[i-1]: seasons_sorted[i] for i in range(1, len(seasons_sorted))}
# la última temporada con estadísticas también describe a la siguiente (p. ej. 2024-2025 -> 2025-2026)
_a = int(seasons_sorted[-1][:4]) + 1
next_of[seasons_sorted[-1]] = f"{_a}-{_a + 1}"

ptf_shifted = ptf.copy()
ptf_shifted['season'] = ptf_shifted['season_label'].map(next_of)
ptf_shifted = ptf_shifted.dropna(subset=['season']).drop(columns='season_label')

home_ptf = ptf_shifted.rename(columns={**{c: f'home_{c}' for c in PLAYER_COLS}, 'team': 'home'})
away_ptf = ptf_shifted.rename(columns={**{c: f'away_{c}' for c in PLAYER_COLS}, 'team': 'away'})

enriched = (
    df
    .merge(home_ptf, on=['season', 'home'], how='left')
    .merge(away_ptf, on=['season', 'away'], how='left')
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURES DIFERENCIALES Y COMPOSITES
# ─────────────────────────────────────────────────────────────────────────────
print("[8] Creando features diferenciales y composites...")

for col in PLAYER_COLS:
    enriched[f'diff_{col}'] = enriched[f'home_{col}'] - enriched[f'away_{col}']

# Índice compuesto de calidad de plantel
enriched['home_player_quality'] = (
    enriched['home_gk_save_pct'].fillna(0) * 0.3 +
    enriched['home_top3_ga_per90'].fillna(0) * 0.4 +
    enriched['home_team_tkl_per90'].fillna(0) * 0.3
)
enriched['away_player_quality'] = (
    enriched['away_gk_save_pct'].fillna(0) * 0.3 +
    enriched['away_top3_ga_per90'].fillna(0) * 0.4 +
    enriched['away_team_tkl_per90'].fillna(0) * 0.3
)
enriched['diff_player_quality'] = enriched['home_player_quality'] - enriched['away_player_quality']

# Features de interacción (mercado + forma)
enriched['market_form_diff']     = enriched['odds_home_prob'] - enriched['diff_roll_points'].clip(-3, 3) / 3 * 0.5
enriched['home_ground_advantage']= enriched['home_roll_home_points'] - enriched['home_roll_away_points']
enriched['diff_defensive']       = enriched['away_roll_goals_scored'] - enriched['home_roll_goals_conceded']
enriched['streak_clash']         = enriched['home_unbeaten_streak'] - enriched['away_unbeaten_streak']
enriched['market_uncertainty']   = enriched['odds_draw_prob'] / (
    enriched['odds_home_prob'] + enriched['odds_away_prob'] + 1e-6)

# ─────────────────────────────────────────────────────────────────────────────
# 9. IMPUTACIÓN DE NULOS (temporadas sin S-1 y equipos recién ascendidos)
# ─────────────────────────────────────────────────────────────────────────────
print("[9] Imputando nulos con mediana de la liga...")

NEW_COLS = (
    [f'home_{c}' for c in PLAYER_COLS] +
    [f'away_{c}' for c in PLAYER_COLS] +
    [f'diff_{c}' for c in PLAYER_COLS] +
    ['home_player_quality', 'away_player_quality', 'diff_player_quality',
     'market_form_diff', 'home_ground_advantage', 'diff_defensive',
     'streak_clash', 'market_uncertainty']
)
n_before = enriched[NEW_COLS].isnull().sum().sum()
# Las medianas de imputación pueden calcularse solo con temporadas hasta IMPUTE_UNTIL_SEASON (p. ej. '2024-2025'),
# para que una temporada de prueba no influya en los valores imputados. Sin la variable se usan todas las filas.
_hasta = os.environ.get('IMPUTE_UNTIL_SEASON')
_ref = enriched if not _hasta else enriched[enriched['season'] <= _hasta]
for col in NEW_COLS:
    med = _ref[col].median()
    enriched[col] = enriched[col].fillna(med)
n_after = enriched[NEW_COLS].isnull().sum().sum()
print(f"  Nulos: {n_before} → {n_after}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. GUARDAR DATASET ENRIQUECIDO
# ─────────────────────────────────────────────────────────────────────────────
enriched.to_csv(f'{D_OUT}/model_dataset_enriched.csv', index=False)
print(f"\n[OK] Guardado: model_dataset_enriched.csv  →  {enriched.shape}")
print(f"  Features totales: {enriched.shape[1]}")
print(f"  Partidos: {enriched.shape[0]}")

# Correlación de las features nuevas con el target
print("\nCorrelación features de jugadores con target_home_win:")
diff_cols = [f'diff_{c}' for c in PLAYER_COLS] + ['diff_player_quality']
corr = enriched[diff_cols + ['target_home_win']].corr()['target_home_win'].drop('target_home_win')
print(corr.sort_values(key=abs, ascending=False).round(4))
print("\nPipeline completado exitosamente.")
