"""
evaluation_utils.py — Funciones de Evaluación de Modelos
=========================================================

Este módulo proporciona funciones para evaluar modelos de clasificación,
generar métricas comparativas y crear gráficos de evaluación.

Objetivo:
    Facilitar la evaluación consistente de modelos a lo largo de
    diferentes experimentos y folds de validación temporal.

Autor: Proyecto de Tesis - Premier League Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from sklearn.calibration import calibration_curve
import json
import os

RUTA_BASELINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resultados", "baseline_mercado.json")


def leer_baseline() -> float:
    """
    Accuracy del baseline del mercado de apuestas, calculado en 01_eda.ipynb y guardado
    en resultados/baseline_mercado.json (así no queda un número escrito a mano en cada notebook).
    """
    if not os.path.exists(RUTA_BASELINE):
        raise FileNotFoundError("Falta resultados/baseline_mercado.json: ejecutar primero 01_eda.ipynb")
    with open(RUTA_BASELINE, encoding="utf-8") as f:
        return float(json.load(f)["accuracy"])


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Modelo"
) -> Dict[str, float]:
    """
    Calcula todas las métricas de evaluación para un modelo de clasificación.
    
    ¿Por qué esta función?
    -----------------------
    En lugar de calcular cada métrica manualmente cada vez, esta función
    centraliza todas las métricas estándar en un solo lugar.
    
    Métricas calculadas:
    - Accuracy: % de predicciones correctas
    - Precision: De los que predije como positivos, ¿cuántos lo son?
    - Recall: De los positivos reales, ¿cuántos detecté?
    - F1-Score: Media armónica de precision y recall
    - ROC-AUC: Capacidad de discriminación (0.5 = aleatorio, 1.0 = perfecto)
    - Brier Score: Calibración de probabilidades (menor = mejor)
    
    Parámetros
    ----------
    y_true : pd.Series
        Valores reales del target (0 o 1)
    y_pred : np.ndarray
        Predicciones del modelo (0 o 1)
    y_proba : np.ndarray
        Probabilidades predichas de la clase positiva
    model_name : str
        Nombre del modelo para identificación
    
    Retorna
    -------
    dict
        Diccionario con todas las métricas calculadas
    
    Ejemplo
    -------
    >>> metrics = evaluate_model(y_test, y_pred, y_proba, "Random Forest")
    >>> print(metrics['f1_score'])
    0.6838
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'brier_score': brier_score_loss(y_true, y_proba),
        'n_samples': len(y_true),
        'positive_rate': y_true.mean()  # Tasa de positivos reales
    }
    
    return metrics


def evaluate_model_per_fold(
    y_true_dict: Dict[int, pd.Series],
    y_pred_dict: Dict[int, np.ndarray],
    y_proba_dict: Dict[int, np.ndarray],
    model_name: str = "Modelo"
) -> pd.DataFrame:
    """
    Evalúa un modelo en múltiples folds y retorna métricas por fold.
    
    ¿Para qué sirve?
    -----------------
    Cuando usamos expanding window, tenemos múltiples folds. Esta función
    calcula métricas para cada fold y luego promedia.
    
    Parámetros
    ----------
    y_true_dict : dict
        {fold_idx: y_true} para cada fold
    y_pred_dict : dict
        {fold_idx: y_pred} para cada fold
    y_proba_dict : dict
        {fold_idx: y_proba} para cada fold
    model_name : str
        Nombre del modelo
    
    Retorna
    -------
    pd.DataFrame
        Tabla con métricas por fold y promedio
    """
    results = []
    
    for fold_idx in sorted(y_true_dict.keys()):
        metrics = evaluate_model(
            y_true_dict[fold_idx],
            y_pred_dict[fold_idx],
            y_proba_dict[fold_idx],
            model_name
        )
        metrics['fold'] = fold_idx + 1
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    
    # Agregar promedio y desviación está.std
    numeric_cols = df_results.select_dtypes(include=[np.number]).columns
    summary = df_results[numeric_cols].mean()
    summary['fold'] = 'Promedio'
    
    df_results = pd.concat([df_results, summary.to_frame().T], ignore_index=True)
    
    return df_results


def compare_models(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Compara múltiples modelos en un DataFrame resumen.
    
    Parámetros
    ----------
    results_dict : dict
        {model_name: metrics_dict} para cada modelo
    metrics : list
        Métricas a incluir (default: ['accuracy', 'f1_score', 'roc_auc', 'brier_score'])
    
    Retorna
    -------
    pd.DataFrame
        Tabla comparativa ordenada por ROC-AUC
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'brier_score']
    
    df = pd.DataFrame(results_dict).T
    df = df[metrics]
    df = df.sort_values('roc_auc', ascending=False)
    
    return df.round(4)


def plot_metric_comparison(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera gráfico de barras comparando múltiples modelos.
    
    ¿Por qué este gráfico?
    -----------------------
    Un gráfico de barras agrupadas permite ver de un vistazo:
    - Qué modelo es mejor en cada métrica
    - Si hay trade-offs entre métricas
    - Si algún modelo es consistentemente mejor
    
    Parámetros
    ----------
    results_dict : dict
        {model_name: metrics_dict}
    metrics : list
        Métricas a graficar
    figsize : tuple
        Tamaño de la figura
    save_path : str
        Ruta para guardar la imagen (opcional)
    
    Retorna
    -------
    matplotlib.figure.Figure
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    df = compare_models(results_dict, metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(metrics))
    n_models = len(df)
    width = 0.8 / n_models
    
    colors = ['#6baed6', '#74c476', '#fd8d3c', '#9e9ac8', '#e07b71', '#bcbd22']
    
    for i, (model_name, row) in enumerate(df.iterrows()):
        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, row[metrics].values, width, 
               label=model_name, color=colors[i % len(colors)],
               edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=11)
    ax.set_ylim(0.5, 0.85)
    baseline = leer_baseline()
    ax.axhline(baseline, color='red', linestyle='--', linewidth=1.2, alpha=0.7, 
               label=f'Baseline mercado ({baseline * 100:.2f}%)')
    ax.set_title('Comparación de Modelos — Métricas Principales', fontsize=13)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylabel('Puntuación')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    
    return fig


def plot_confusion_matrices(
    y_true_dict: Dict[int, pd.Series],
    y_pred_dict: Dict[int, np.ndarray],
    model_names: List[str],
    figsize: Tuple[int, int] = (16, 9),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera matrices de confusión para múltiples modelos.
    
    ¿Para qué sirve?
    -----------------
    La matriz de confusión muestra:
    - True Negatives: Predije "no victoria" y era "no victoria"
    - False Positives: Predije "victoria" y era "no victoria" (error tipo I)
    - False Negatives: Predije "no victoria" y era "victoria" (error tipo II)
    - True Positives: Predije "victoria" y era "victoria"
    
    Parámetros
    ----------
    y_true_dict : dict
        {model_name: y_true} (o {fold_idx: y_true} si es un solo modelo)
    y_pred_dict : dict
        {model_name: y_pred}
    model_names : list
        Nombres de los modelos
    figsize : tuple
        Tamaño de la figura
    save_path : str
        Ruta para guardar
    
    Retorna
    -------
    matplotlib.figure.Figure
    """
    n_models = len(model_names)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        cm = confusion_matrix(y_true_dict[i], y_pred_dict[i])
        ConfusionMatrixDisplay(cm, display_labels=['No victoria', 'Victoria']).plot(
            ax=axes[i], cmap='Blues', colorbar=False
        )
        
        # Calcular métricas adicionales
        tn, fp, fn, tp = cm.ravel()
        total = cm.sum()
        
        axes[i].set_title(f'{model_name}\n'
                         f'VN={tn} FP={fp} | FN={fn} VP={tp}', fontsize=9)
    
    # Ocultar ejes vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Matrices de Confusión — Todos los Modelos', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    
    return fig


def plot_roc_curves(
    y_true: pd.Series,
    y_proba_dict: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera curvas ROC para múltiples modelos.
    
    ¿Qué es la curva ROC?
    ----------------------
    La curva ROC (Receiver Operating Characteristic) muestra:
    - Eje X: Tasa de Falsos Positivos (FPR)
    - Eje Y: Tasa de Verdaderos Positivos (TPR)
    
    Un modelo perfecto estaría en la esquina superior izquierda.
    Un modelo aleatorio sería una diagonal (AUC = 0.5).
    
    El AUC (Area Under Curve) resume la curva:
    - AUC = 0.5: Modelo aleatorio
    - AUC = 0.7-0.8: Aceptable
    - AUC = 0.8-0.9: Bueno
    - AUC > 0.9: Excelente
    
    Parámetros
    ----------
    y_true : pd.Series
        Valores reales
    y_proba_dict : dict
        {model_name: y_proba}
    figsize : tuple
        Tamaño de la figura
    save_path : str
        Ruta para guardar
    
    Retorna
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#6baed6', '#74c476', '#fd8d3c', '#9e9ac8', '#e07b71', '#bcbd22']
    
    for i, (model_name, y_proba) in enumerate(y_proba_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f'{model_name} (AUC={auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatorio (AUC=0.50)')
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
    ax.set_title('Curvas ROC — Comparación de Modelos', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    
    return fig


def plot_calibration_curves(
    y_true: pd.Series,
    y_proba_dict: Dict[str, np.ndarray],
    n_bins: int = 10,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera curvas de calibración de probabilidades.
    
    ¿Qué es calibración?
    ---------------------
    Un modelo está calibrado si:
    - Cuando predice 70% de probabilidad de victoria
    - Realmente gana ~70% de esas veces
    
    Si no está calibrado:
    - Puede sobreestimar (predice 80% pero solo gana 60%)
    - Puede subestimar (predice 40% pero gana 60%)
    
    El Brier Score mide esto:
    - Brier = 0: Calibración perfecta
    - Brier = 0.25: Modelo aleatorio
    - Brier > 0.25: Peor que aleatorio
    
    Parámetros
    ----------
    y_true : pd.Series
        Valores reales
    y_proba_dict : dict
        {model_name: y_proba}
    n_bins : int
        Número de bins para agrupar probabilidades
    figsize : tuple
        Tamaño
    save_path : str
        Ruta para guardar
    
    Retorna
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Calibración perfecta')
    
    colors = ['#6baed6', '#74c476', '#fd8d3c', '#9e9ac8', '#e07b71', '#bcbd22']
    
    for i, (model_name, y_proba) in enumerate(y_proba_dict.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        brier = brier_score_loss(y_true, y_proba)
        
        ax.plot(prob_pred, prob_true, 's-', color=colors[i % len(colors)],
                linewidth=2, markersize=5,
                label=f'{model_name} (Brier={brier:.4f})')
    
    ax.set_xlabel('Probabilidad media predicha')
    ax.set_ylabel('Fracción de positivos reales')
    ax.set_title('Curvas de Calibración — Todos los Modelos', fontsize=12)
    ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    
    return fig


def plot_feature_importance(
    importances: Dict[str, pd.Series],
    top_n: int = 20,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera gráficos de importancia de features para modelos de árboles.
    
    ¿Por qué importa?
    ------------------
    La importancia de features nos dice:
    - Qué variables son más predictive
    - Cuáles se pueden eliminar sin perder rendimiento
    - Si hay features irrelevantes que ruidan el modelo
    
    Parámetros
    ----------
    importances : dict
        {model_name: feature_importances_series}
    top_n : int
        Número de features a mostrar
    figsize : tuple
        Tamaño
    save_path : str
        Ruta para guardar
    
    Retorna
    -------
    matplotlib.figure.Figure
    """
    n_models = len(importances)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    colors_map = {
        'mercado': '#6baed6',
        'racha': '#fd8d3c',
        'forma': '#74c476',
        'jugadores': '#e07b71',
        'otro': '#9e9ac8'
    }
    
    def get_color(feature_name):
        if any(p in feature_name for p in ['odds_', 'market_']):
            return colors_map['mercado']
        elif 'streak' in feature_name:
            return colors_map['racha']
        elif 'roll' in feature_name:
            return colors_map['forma']
        elif any(p in feature_name for p in ['gk_', 'squad_', 'tkl_', 'int_', 'discipline', 'on_off', 'player_quality']):
            return colors_map['jugadores']
        return colors_map['otro']
    
    for ax, (model_name, imp) in zip(axes, importances.items()):
        top_features = imp.sort_values(ascending=False).head(top_n).sort_values()
        
        colors = [get_color(f) for f in top_features.index]
        top_features.plot(kind='barh', color=colors, edgecolor='white', ax=ax)
        
        ax.set_title(f'Top {top_n} Features — {model_name}', fontsize=11)
        ax.set_xlabel('Importancia')
        
        # Agregar valores
        for bar in ax.patches:
            width = bar.get_width()
            if width > 0.001:
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{width:.4f}', va='center', fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    
    return fig


def print_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Modelo"
) -> None:
    """
    Imprime un reporte de clasificación formateado.
    
    Parámetros
    ----------
    y_true : pd.Series
        Valores reales
    y_pred : np.ndarray
        Predicciones
    model_name : str
        Nombre del modelo
    """
    print(f"\n{'='*60}")
    print(f"REPORTE DE CLASIFICACIÓN — {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, 
                               target_names=['No victoria', 'Victoria']))


def save_results(
    results_dict: Dict[str, Dict],
    filepath: str,
    index: bool = True
) -> None:
    """
    Guarda resultados en un archivo CSV.
    
    Parámetros
    ----------
    results_dict : dict
        Diccionario con resultados
    filepath : str
        Ruta del archivo CSV
    index : bool
        Si incluir el índice
    """
    df = pd.DataFrame(results_dict).T if not isinstance(results_dict, pd.DataFrame) else results_dict
    df.to_csv(filepath, index=index)
    print(f"Resultados guardados en: {filepath}")


if __name__ == "__main__":
    print("Módulo de evaluación de modelos.")
    print("Importar funciones con: from utils.evaluation_utils import evaluate_model")
