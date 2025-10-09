# src/evaluate.py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

def auroc_auprc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)

def find_threshold_for_specificity(y_true, y_prob, target_spec=0.95):
    ths = np.linspace(0, 1, 1001)
    best_t, best_diff = 0.5, 1e9
    for t in ths:
        yhat = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
        spec = tn / (tn + fp + 1e-8)
        d = abs(spec - target_spec)
        if d < best_diff:
            best_t, best_diff = t, d
    return best_t

def specificity_sensitivity(y_true, y_prob, thr):
    yhat = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
    spec = tn / (tn + fp + 1e-8)
    sens = tp / (tp + fn + 1e-8)
    return spec, sens
