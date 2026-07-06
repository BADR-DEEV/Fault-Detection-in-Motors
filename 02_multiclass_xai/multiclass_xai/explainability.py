"""
Explainability utilities for physics-aligned MaFaulDa fault diagnosis.
Optional SHAP and LIME dependencies are imported only when their functions run.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance


logger = logging.getLogger(__name__)


def compute_permutation_feature_importance(
    model,
    X_test,
    y_test,
    feature_names,
    class_names,
    n_repeats=10,
    random_state=42,
    save_path="feature_importance_physics_validated.png",
):
    """Compute and plot permutation feature importance using macro-F1."""
    logger.info("Computing permutation feature importance (F1-macro, leakage-proof).")

    perm_imp = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="f1_macro",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm_imp.importances_mean,
            "importance_std": perm_imp.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    physics_interpretation = {
        "rad_kurt": {"fault": "Bearing", "physics": "Impulsive impacts -> high kurtosis"},
        "ax_kurt": {"fault": "Bearing", "physics": "Axial bearing impacts"},
        "tan_kurt": {"fault": "Bearing", "physics": "Tangential bearing impacts"},
        "rad_spec_centroid": {"fault": "Imbalance", "physics": "Energy at 1x RPM"},
        "ax_spec_spread": {"fault": "Misalignment", "physics": "Harmonic-rich spectrum"},
        "rad_spec_spread": {"fault": "Misalignment", "physics": "Radial coupling dynamics"},
        "axial_ratio": {"fault": "Horiz_Misalign", "physics": "Axial energy concentration"},
        "radial_ratio": {"fault": "Imbalance", "physics": "Centrifugal force dominance"},
        "rad_rms": {"fault": "General", "physics": "Overall vibration severity"},
        "rad_crest": {"fault": "Bearing", "physics": "Peak-to-RMS amplifies shocks"},
        "ax_rms": {"fault": "Misalignment", "physics": "Axial coupling forces"},
        "rad_freq_median": {"fault": "Bearing", "physics": "Resonance shifts median frequency"},
        "rad_freq_rolloff85": {"fault": "Bearing", "physics": "High-frequency resonance content"},
        "ax_crest": {"fault": "Misalignment", "physics": "Axial shock events"},
        "tan_rms": {"fault": "General", "physics": "Weakest axis in bearing geometry"},
    }

    logger.info("%s", "=" * 70)
    logger.info("Top features and physics interpretation")
    logger.info("%s", "=" * 70)
    top_n = min(15, len(importance_df))
    for index in range(top_n):
        feature = importance_df.loc[index, "feature"]
        mean_imp = importance_df.loc[index, "importance_mean"]
        std_imp = importance_df.loc[index, "importance_std"]
        interp = physics_interpretation.get(feature, {"fault": "Unknown", "physics": "No interpretation"})
        logger.info(
            "%2d. %-20s | dF1 = %.4f +/- %.4f | %s: %s",
            index + 1,
            feature,
            mean_imp,
            std_imp,
            interp["fault"],
            interp["physics"],
        )

    colors = []
    for feature in importance_df["feature"][:top_n]:
        if "kurt" in feature or "crest" in feature:
            colors.append("#8e44ad")
        elif "spec_centroid" in feature or "radial_ratio" in feature:
            colors.append("#e74c3c")
        elif "spec_spread" in feature or "axial_ratio" in feature:
            colors.append("#e67e22")
        else:
            colors.append("#3498db")

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(top_n)
    plt.barh(
        y_pos,
        importance_df["importance_mean"][:top_n][::-1],
        xerr=importance_df["importance_std"][:top_n][::-1],
        color=colors[::-1],
        alpha=0.85,
        capsize=4,
    )
    plt.yticks(y_pos, importance_df["feature"][:top_n][::-1], fontsize=10)
    plt.xlabel("Permutation Importance (delta F1-macro)", fontsize=12, fontweight="bold")
    plt.title("Physics-Aligned Feature Importance\n(Test Set, Leakage-Proof)", fontsize=14, fontweight="bold")
    plt.grid(axis="x", alpha=0.3, linestyle="--")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#8e44ad", label="Bearing Faults"),
        Patch(facecolor="#e74c3c", label="Imbalance"),
        Patch(facecolor="#e67e22", label="Misalignment"),
        Patch(facecolor="#3498db", label="General Features"),
    ]
    plt.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    csv_path = save_path.with_suffix(".csv")
    importance_df.to_csv(csv_path, index=False)
    logger.info("Saved PFI plot to: %s", save_path)
    logger.info("Saved PFI CSV to: %s", csv_path)

    return perm_imp, importance_df


def explain_with_shap(model, X_train, X_test, feature_names, class_names, max_display=15, output_dir="."):
    """Generate SHAP explanations with KernelExplainer for probability models."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP is not installed. Skipping SHAP explanations.")
        return None, None

    logger.info("Generating SHAP explanations.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    X_test_sample = X_test[:200] if len(X_test) > 200 else X_test
    shap_values = explainer.shap_values(X_test_sample)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test_sample,
        feature_names=feature_names,
        class_names=class_names,
        max_display=max_display,
        show=False,
    )
    plt.suptitle("SHAP Summary: Global Feature Impact", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test_sample,
        feature_names=feature_names,
        class_names=class_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.suptitle("SHAP Mean Absolute Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    return shap_values, explainer


def explain_with_lime(model, X_instance, X_train, feature_names, class_names, num_features=10, output_dir="."):
    """Generate a LIME explanation for one instance."""
    try:
        import lime.lime_tabular
    except ImportError:
        logger.warning("LIME is not installed. Skipping LIME explanation.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_proba = model.predict_proba(X_instance.reshape(1, -1))[0]
    pred_class = int(np.argmax(pred_proba))

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
    )
    exp = explainer.explain_instance(
        X_instance,
        model.predict_proba,
        num_features=num_features,
        labels=[pred_class],
    )

    logger.info("LIME explanation for: %s", class_names[pred_class])
    for feature, weight in exp.as_list(label=pred_class):
        direction = "increases probability" if weight > 0 else "decreases probability"
        logger.info("  %s: %+0.4f (%s)", feature, weight, direction)

    fig = exp.as_pyplot_figure(label=pred_class)
    fig.suptitle(f"LIME: {class_names[pred_class]}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "lime_explanation.png", dpi=300, bbox_inches="tight")
    plt.close()
    return exp


def plot_partial_dependence(model, X_test, feature_names, class_names, top_features=None, n_cols=3, output_path="partial_dependence.png"):
    """Plot partial dependence for top features across each class."""
    logger.info("Generating partial dependence plots.")

    if top_features is None:
        feature_var = np.var(X_test, axis=0)
        top_idx = np.argsort(feature_var)[-9:]
        top_features = [feature_names[index] for index in top_idx]

    feature_indices = [feature_names.index(feature) for feature in top_features[:9]]
    n_classes = len(class_names)
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = [axes] if n_rows * n_cols == 1 else axes.flatten()

    for index, class_name in enumerate(class_names):
        if index < len(axes):
            PartialDependenceDisplay.from_estimator(
                model,
                X_test,
                features=feature_indices,
                feature_names=feature_names,
                target=index,
                ax=axes[index],
                line_kw={"label": class_name},
            )
            axes[index].set_title(class_name, fontweight="bold")
            axes[index].legend(fontsize=8)

    for index in range(len(class_names), len(axes)):
        fig.delaxes(axes[index])

    plt.suptitle(
        "Partial Dependence: Feature Impact on Class Probabilities\n"
        "(Non-linear relationships reveal physics mechanisms)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved partial dependence plot to: %s", output_path)


def run_full_explainability_suite(
    model,
    X_train,
    X_test,
    y_test,
    feature_names,
    class_names,
    instance_idx=0,
    run_shap=True,
    run_lime=True,
    run_pdp=True,
    run_pfi=True,
    output_dir=".",
):
    """Run the available explainability methods."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_pfi:
        compute_permutation_feature_importance(
            model,
            X_test,
            y_test,
            feature_names,
            class_names,
            save_path=output_dir / "feature_importance_physics_validated.png",
        )
    if run_shap:
        explain_with_shap(model, X_train, X_test, feature_names, class_names, output_dir=output_dir)
    if run_lime and len(X_test) > instance_idx:
        explain_with_lime(model, X_test[instance_idx], X_train, feature_names, class_names, output_dir=output_dir)
    if run_pdp:
        plot_partial_dependence(model, X_test, feature_names, class_names, output_path=output_dir / "partial_dependence.png")
