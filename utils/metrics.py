import numpy as np
from sklearn.metrics import confusion_matrix


def cost_score(y_true, y_pred, amounts, fn_multiplier=5.0, fp_multiplier=0.5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    amounts = np.array(amounts)
    amounts_score = np.log1p(amounts)

    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)

    fn_cost = amounts_score[fn_mask].sum() * fn_multiplier
    fp_cost = amounts_score[fp_mask].sum() * fp_multiplier

    return fn_cost + fp_cost


def cost_curve(y_true, y_pred_proba, amounts, **multipliers):
    thresholds = np.linspace(0.005, 1.0, 200)
    costs = []

    for thr in thresholds:
        y_pred = (y_pred_proba >= thr).astype(int)
        cost = cost_score(y_true, y_pred, amounts, **multipliers)
        costs.append(cost)

    return thresholds, costs


def confusion_curve(y_true, y_pred_proba):
    thresholds = np.linspace(0.005, 1.0, 200)
    confusions = []

    for thr in thresholds:
        y_pred = (y_pred_proba >= thr).astype(int)
        confusion = confusion_matrix(y_true, y_pred)
        confusions.append(confusion)

    confusions = np.array(confusions)

    tn = confusions[:, 0, 0]
    fp = confusions[:, 0, 1]
    fn = confusions[:, 1, 0]
    tp = confusions[:, 1, 1]

    return thresholds, (tn, fp, fn, tp)
