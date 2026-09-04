"""SHAP interpretation for the champion model (LightGBM tree model or logistic WoE scorecard)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
SHAP_SUMMARY_PATH = OUTPUT_DIR / "shap_summary.png"


def _positive_class_shap_values(shap_values):
    """Normalize shap_values across shap-library versions to a single (n, features) array
    for the positive (bad/default) class."""
    if isinstance(shap_values, list):
        return shap_values[1]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def explain_champion(model, X_sample, model_kind: str, out_path: Path = SHAP_SUMMARY_PATH) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if model_kind == "lightgbm":
        explainer = shap.TreeExplainer(model)
        shap_values = _positive_class_shap_values(explainer.shap_values(X_sample))
    else:
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
