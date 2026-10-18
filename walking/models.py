"""Model definitions shared by the sanity check and the active learning runs:
median imputer + standard scaler + balanced logistic regression (C=1)."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C = 1.0


def make_clf():
    return LogisticRegression(max_iter=2000, C=C, class_weight="balanced")


def make_prep():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


def make_pipe():
    """Full pipeline: preprocessing fit on the training rows + classifier."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), make_clf())


def fit_evaluate(X_lab, y_lab, X_te, y_te):
    """Fit the classifier on (X_lab, y_lab); return (metrics, fitted clf). Metrics: train/test
    AUROC and test AUPRC. Inputs are expected to be preprocessed already (see make_prep)."""
    clf = make_clf().fit(X_lab, y_lab)
    p_te = clf.predict_proba(X_te)[:, 1]
    p_tr = clf.predict_proba(X_lab)[:, 1]
    return dict(test_auroc=roc_auc_score(y_te, p_te), test_auprc=average_precision_score(y_te, p_te),
                train_auroc=roc_auc_score(y_lab, p_tr) if 0 < y_lab.sum() < len(y_lab) else np.nan), clf


def evaluate(X_lab, y_lab, X_te, y_te):
    return fit_evaluate(X_lab, y_lab, X_te, y_te)[0]
