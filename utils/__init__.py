"""
utils — Paquete de Utilidades para el Proyecto de Tesis
========================================================

Este paquete contiene módulos utilitarios para el pipeline de
machine learning del proyecto de predicción de la Premier League.

Módulos
-------
- split_utils: Funciones de división temporal de datos
- evaluation_utils: Funciones de evaluación de modelos

Uso
---
from utils.split_utils import expanding_window_split
from utils.evaluation_utils import evaluate_model
"""

from .split_utils import (
    expanding_window_split,
    get_fold_info,
    expanding_window_with_cv,
    temporal_train_test_split,
    print_fold_summary
)

from .evaluation_utils import (
    evaluate_model,
    evaluate_model_per_fold,
    compare_models,
    plot_metric_comparison,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_calibration_curves,
    plot_feature_importance,
    print_classification_report,
    save_results,
    leer_baseline
)

__all__ = [
    # Split utils
    'expanding_window_split',
    'get_fold_info',
    'expanding_window_with_cv',
    'temporal_train_test_split',
    'print_fold_summary',
    # Evaluation utils
    'evaluate_model',
    'evaluate_model_per_fold',
    'compare_models',
    'plot_metric_comparison',
    'plot_confusion_matrices',
    'plot_roc_curves',
    'plot_calibration_curves',
    'plot_feature_importance',
    'print_classification_report',
    'save_results',
    'leer_baseline'
]
