"""
split_utils.py — Funciones de División Temporal de Datos
==========================================================

Este módulo proporciona funciones para dividir datos de series temporales
de manera que se respete el orden cronológico, evitando data leakage.

Objetivo:
    Simular cómo funcionaría el modelo en producción, donde siempre
    entrenamos con datos del pasado y predecimos el futuro.

Autor: Proyecto de Tesis - Premier League Prediction
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def expanding_window_split(
    df: pd.DataFrame,
    season_col: str = 'season',
    min_train_seasons: int = 1,
    test_size: int = 1
) -> List[Tuple[List[int], List[int], List[str], str]]:
    """
    Genera folds usando expanding window (ventana creciente).
    
    ¿Qué es expanding window?
    --------------------------
    Es una estrategia de validación temporal donde:
    - Fold 1: Entrenamos con 1 temporada, testeamos con la siguiente
    - Fold 2: Entrenamos con 2 temporadas, testeamos con la siguiente
    - Fold 3: Entrenamos con 3 temporadas, testeamos con la siguiente
    - ... y así sucesivamente
    
    Ventaja sobre split simple:
    - El modelo se entrena con cada vez más datos
    - Simula mejor la realidad (en producción, siempre tenemos más historia)
    - Permite evaluar la estabilidad del modelo a lo largo del tiempo
    
    ¿Por qué no usar train_test_split aleatorio?
    ---------------------------------------------
    Porque los datos de fútbol son temporales. Si mezclamos épocas,
    el modelo "ve el futuro" y las métricas serían artificialmente altas.
    
    Parámetros
    ----------
    df : pd.DataFrame
        Dataset completo con los datos
    season_col : str
        Nombre de la columna que identifica la temporada (default: 'season')
    min_train_seasons : int
        Número mínimo de temporadas de entrenamiento (default: 1)
    test_size : int
        Número de temporadas de test en cada fold (default: 1)
    
    Retorna
    -------
    list of tuples
        Cada tupla contiene:
        - train_idx: Índices del set de entrenamiento
        - test_idx: Índices del set de test
        - train_seasons: Lista de temporadas de entrenamiento
        - test_season: Temporada de test
    
    Ejemplo
    -------
    Para 8 temporadas (2017-2025) con min_train_seasons=1:
    
    Fold 1: Train [2017-2018]           → Test 2018-2019
    Fold 2: Train [2017-2019]           → Test 2019-2020
    Fold 3: Train [2017-2020]           → Test 2020-2021
    Fold 4: Train [2017-2021]           → Test 2021-2022
    Fold 5: Train [2017-2022]           → Test 2022-2023
    Fold 6: Train [2017-2023]           → Test 2023-2024
    Fold 7: Train [2017-2024]           → Test 2024-2025
    """
    # Obtener temporadas ordenadas cronológicamente
    seasons = sorted(df[season_col].unique())
    
    # Validar que hay suficientes temporadas
    if len(seasons) < min_train_seasons + test_size:
        raise ValueError(
            f"Se necesitan al menos {min_train_seasons + test_size} temporadas, "
            f"pero solo hay {len(seasons)}"
        )
    
    folds = []
    
    # Generar folds: desde min_train_seasons hasta (total - test_size)
    for i in range(min_train_seasons, len(seasons) - test_size + 1):
        # Temporadas de entrenamiento: desde la primera hasta la i-ésima
        train_seasons = seasons[:i]
        
        # Temporada(s) de test: la(s) siguiente(s)
        test_seasons = seasons[i:i + test_size]
        
        # Obtener índices del DataFrame
        train_mask = df[season_col].isin(train_seasons)
        test_mask = df[season_col].isin(test_seasons)
        
        train_idx = df[train_mask].index.tolist()
        test_idx = df[test_mask].index.tolist()
        
        folds.append((
            train_idx,
            test_idx,
            train_seasons,
            test_seasons if test_size > 1 else test_seasons[0]
        ))
    
    return folds


def get_fold_info(
    df: pd.DataFrame,
    folds: list,
    season_col: str = 'season'
) -> pd.DataFrame:
    """
    Genera un DataFrame con información resumida de cada fold.
    
    Útil para documentar y entender la distribución de datos
    en cada iteración de la validación temporal.
    
    Parámetros
    ----------
    df : pd.DataFrame
        Dataset completo
    folds : list
        Lista de folds generada por expanding_window_split
    season_col : str
        Nombre de la columna de temporada
    
    Retorna
    -------
    pd.DataFrame
        Tabla con información de cada fold:
        - fold: Número de fold
        - train_seasons: Temporadas de entrenamiento
        - test_season: Temporada de test
        - n_train: Número de partidos en train
        - n_test: Número de partidos en test
        - train_ratio: Proporción de datos de entrenamiento
    """
    info = []
    
    for i, (train_idx, test_idx, train_s, test_s) in enumerate(folds):
        n_train = len(train_idx)
        n_test = len(test_idx)
        total = n_train + n_test
        
        info.append({
            'fold': i + 1,
            'train_seasons': str(train_s) if isinstance(train_s, list) else train_s,
            'test_season': str(test_s) if isinstance(test_s, list) else test_s,
            'n_train': n_train,
            'n_test': n_test,
            'train_ratio': round(n_train / total, 3)
        })
    
    return pd.DataFrame(info)


def expanding_window_with_cv(
    df: pd.DataFrame,
    season_col: str = 'season',
    n_inner_splits: int = 3
) -> List[Tuple]:
    """
    Expanding window con validación cruzada interna para GridSearchCV.
    
    ¿Para qué sirve?
    -----------------
    Cuando usamos GridSearchCV, necesitamos un cv splitter interno.
    Este función genera splits internos que también respetan el tiempo.
    
    Ejemplo de uso con GridSearchCV:
    
        inner_cv = expanding_window_with_cv(X_train, n_inner_splits=3)
        gs = GridSearchCV(model, param_grid, cv=inner_cv)
        gs.fit(X_train, y_train)
    
    Parámetros
    ----------
    df : pd.DataFrame
        Datos de entrenamiento (NO el dataset completo)
    season_col : str
        Columna de temporada
    n_inner_splits : int
        Número de splits internos para la validación cruzada
    
    Retorna
    -------
    list of tuples
        Cada tupla es (train_idx, test_idx) para el CV interno
    """
    seasons = sorted(df[season_col].unique())
    n_seasons = len(seasons)
    
    if n_seasons < n_inner_splits + 1:
        # Si no hay suficientes temporadas, usar Leave-One-Out
        n_inner_splits = n_seasons - 1
    
    inner_folds = []
    
    for i in range(1, n_inner_splits + 1):
        # Dividir las temporadas disponibles en train/test interno
        split_point = int(n_seasons * i / (n_inner_splits + 1))
        
        train_seasons = seasons[:split_point]
        test_season = seasons[split_point]
        
        train_mask = df[season_col].isin(train_seasons)
        test_mask = df[season_col] == test_season
        
        inner_folds.append((
            df[train_mask].index.tolist(),
            df[test_mask].index.tolist()
        ))
    
    return inner_folds


def temporal_train_test_split(
    df: pd.DataFrame,
    season_col: str = 'season',
    test_seasons: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split simple temporal: las últimas N temporadas como test.
    
    Alternativa al split aleatorio. Útil para una evaluación rápida
    sin la complejidad del expanding window.
    
    Parámetros
    ----------
    df : pd.DataFrame
        Dataset completo
    season_col : str
        Columna de temporada
    test_seasons : int
        Número de temporadas de test (últimas N temporadas)
    
    Retorna
    -------
    tuple of DataFrames
        (df_train, df_test)
    """
    seasons = sorted(df[season_col].unique())
    
    train_seasons = seasons[:-test_seasons]
    test_season = seasons[-test_seasons:]
    
    df_train = df[df[season_col].isin(train_seasons)].copy()
    df_test = df[df[season_col].isin(test_season)].copy()
    
    return df_train, df_test


def print_fold_summary(folds: list, df: pd.DataFrame) -> None:
    """
    Imprime un resumen legible de los folds.
    
    Parámetros
    ----------
    folds : list
        Lista de folds generada por expanding_window_split
    df : pd.DataFrame
        Dataset completo (para calcular porcentajes)
    """
    total = len(df)
    
    print("=" * 70)
    print("RESUMEN DE FOLDS — EXPANDING WINDOW")
    print("=" * 70)
    print(f"Total de partidos: {total}")
    print(f"Total de folds: {len(folds)}")
    print("=" * 70)
    
    for i, (train_idx, test_idx, train_s, test_s) in enumerate(folds):
        n_train = len(train_idx)
        n_test = len(test_idx)
        
        print(f"\nFold {i+1}:")
        print(f"  Train: {train_s}")
        print(f"         {n_train} partidos ({n_train/total*100:.1f}%)")
        print(f"  Test:  {test_s}")
        print(f"         {n_test} partidos ({n_test/total*100:.1f}%)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Ejemplo de uso
    print("Módulo de utilidades para división temporal de datos.")
    print("Importar funciones con: from utils.split_utils import expanding_window_split")
