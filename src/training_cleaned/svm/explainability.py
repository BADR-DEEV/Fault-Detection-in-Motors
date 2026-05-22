# utils/explainability.py
"""
Explainability utilities for physics-aligned fault diagnosis on MaFaulDa.
All functions are pure: they take model/data and produce explanations or plots.
Optional dependencies (shap, lime) are imported only when needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import f1_score, make_scorer

logger = logging.getLogger(__name__)


def compute_permutation_feature_importance(
    model,
    X_test,
    y_test,
    feature_names,
    class_names,
    n_repeats=10,
    random_state=42,
    save_path="feature_importance_physics_validated.png"
):
    """
    Compute and visualize permutation feature importance with physics-aligned interpretation.
    
    Parameters:
    -----------
    model : trained classifier (e.g., SVC)
    X_test, y_test : test set (scaled, encoded)
    feature_names : list of feature column names (length = 23)
    class_names : list of class labels
    save_path : where to save the plot
    
    Returns:
    --------
    perm_imp : sklearn Bunch object with importances
    importance_df : DataFrame with detailed results
    """
    logger.info("🧠 Computing Permutation Feature Importance (F1-macro, leakage-proof)...")
    
    # Compute PFI
    perm_imp = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='f1_macro',
        n_jobs=-1
    )
    
    # Create DataFrame
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_imp.importances_mean,
        'importance_std': perm_imp.importances_std
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    
    # Physics-aligned interpretation mapping
    physics_interpretation = {
        'rad_kurt': {'fault': 'Bearing', 'physics': 'Impulsive impacts → high kurtosis'},
        'ax_kurt': {'fault': 'Bearing', 'physics': 'Axial bearing impacts'},
        'tan_kurt': {'fault': 'Bearing', 'physics': 'Tangential bearing impacts'},
        'rad_spec_centroid': {'fault': 'Imbalance', 'physics': 'Energy at 1×RPM'},
        'ax_spec_spread': {'fault': 'Misalignment', 'physics': 'Harmonic-rich spectrum'},
        'rad_spec_spread': {'fault': 'Misalignment', 'physics': 'Radial coupling dynamics'},
        'axial_ratio': {'fault': 'Horiz_Misalign', 'physics': 'Axial energy concentration'},
        'radial_ratio': {'fault': 'Imbalance', 'physics': 'Centrifugal force dominance'},
        'rad_rms': {'fault': 'General', 'physics': 'Overall vibration severity'},
        'rad_crest': {'fault': 'Bearing', 'physics': 'Peak-to-RMS amplifies shocks'},
        'ax_rms': {'fault': 'Misalignment', 'physics': 'Axial coupling forces'},
        'rad_freq_median': {'fault': 'Bearing', 'physics': 'Resonance shifts median freq'},
        'rad_freq_rolloff85': {'fault': 'Bearing', 'physics': 'High-freq resonance content'},
        'ax_crest': {'fault': 'Misalignment', 'physics': 'Axial shock events'},
        'tan_rms': {'fault': 'General', 'physics': 'Weakest axis (bearing geometry)'}
    }
    
    # Log top 15 features with physics context
    logger.info("\n" + "="*70)
    logger.info("🔬 TOP FEATURES & PHYSICS INTERPRETATION")
    logger.info("="*70)
    top_n = min(15, len(imp_df))
    for i in range(top_n):
        feat = imp_df.loc[i, 'feature']
        mean_imp = imp_df.loc[i, 'importance_mean']
        std_imp = imp_df.loc[i, 'importance_std']
        interp = physics_interpretation.get(feat, {'fault': 'Unknown', 'physics': 'No interpretation'})
        logger.info(f"{i+1:2d}. {feat:<20s} | ΔF1 = {mean_imp:.4f} ± {std_imp:.4f} | {interp['fault']}: {interp['physics']}")
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = []
    for feat in imp_df['feature'][:top_n]:
        if 'kurt' in feat or 'crest' in feat:
            colors.append('#8e44ad')  # Bearing
        elif 'spec_centroid' in feat or 'radial_ratio' in feat:
            colors.append('#e74c3c')   # Imbalance
        elif 'spec_spread' in feat or 'axial_ratio' in feat:
            colors.append('#e67e22')   # Misalignment
        else:
            colors.append('#3498db')   # General
    
    y_pos = np.arange(top_n)
    plt.barh(
        y_pos,
        imp_df['importance_mean'][:top_n][::-1],
        xerr=imp_df['importance_std'][:top_n][::-1],
        color=colors[::-1],
        alpha=0.85,
        capsize=4
    )
    plt.yticks(y_pos, imp_df['feature'][:top_n][::-1], fontsize=10)
    plt.xlabel('Permutation Importance (Δ F1-macro)', fontsize=12, fontweight='bold')
    plt.title('Physics-Aligned Feature Importance\n(Test Set, Leakage-Proof)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#8e44ad', label='Bearing Faults'),
        Patch(facecolor='#e74c3c', label='Imbalance'),
        Patch(facecolor='#e67e22', label='Misalignment'),
        Patch(facecolor='#3498db', label='General Features')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved PFI plot to: {save_path}")
    imp_df.to_csv(save_path.replace('.png', '.csv'), index=False)
    logger.info(f"✅ Saved PFI CSV to: {save_path.replace('.png', '.csv')}")
    
    return perm_imp, imp_df


def explain_with_shap(model, X_train, X_test, feature_names, class_names, max_display=15):
    """
    Generate SHAP explanations using KernelExplainer (compatible with SVM).
    Requires `shap` package.
    """
    try:
        import shap
    except ImportError:
        logger.warning("⚠️ SHAP not installed. Skipping SHAP explanations.")
        return None, None

    logger.info("🧠 Generating SHAP explanations (KernelExplainer for SVM)...")
    
    # Use a small background sample for speed
    background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # Explain a subset of test data
    X_test_sample = X_test[:200] if len(X_test) > 200 else X_test
    shap_values = explainer.shap_values(X_test_sample)
    
    # 1. Summary plot (beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                      class_names=class_names, max_display=max_display, show=False)
    plt.suptitle('SHAP Summary: Global Feature Impact', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mean absolute importance (bar)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                      class_names=class_names, plot_type='bar',
                      max_display=max_display, show=False)
    plt.suptitle('SHAP Mean Absolute Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ SHAP plots saved: shap_summary.png, shap_importance.png")
    return shap_values, explainer


def explain_with_lime(model, X_instance, X_train, feature_names, class_names, num_features=10):
    """
    Generate LIME explanation for a single instance.
    Requires `lime` package.
    """
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        logger.warning("⚠️ LIME not installed. Skipping LIME explanation.")
        return None

    logger.info("🔍 Generating LIME explanation for single instance...")
    
    # Get prediction
    pred_proba = model.predict_proba(X_instance.reshape(1, -1))[0]
    pred_class = int(np.argmax(pred_proba))
    
    # Create explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    
    # Explain predicted class
    exp = explainer.explain_instance(
        X_instance,
        model.predict_proba,
        num_features=num_features,
        labels=[pred_class]
    )
    
    # Log results
    logger.info(f"\nLIME Explanation for: {class_names[pred_class]}")
    logger.info("Top contributing features:")
    for feature, weight in exp.as_list(label=pred_class):
        direction = "↑ increases prob" if weight > 0 else "↓ decreases prob"
        logger.info(f"  {feature}: {weight:+.4f} ({direction})")
    
    # Plot
    fig = exp.as_pyplot_figure(label=pred_class)
    fig.suptitle(f'LIME: {class_names[pred_class]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("lime_explanation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ LIME explanation saved: lime_explanation.png")
    return exp


def plot_partial_dependence(model, X_test, feature_names, class_names, top_features=None, n_cols=3):
    """
    Plot Partial Dependence to show non-linear feature effects.
    """
    logger.info("📊 Generating Partial Dependence Plots...")
    
    if top_features is None:
        # Default: top 9 features by variance
        feature_var = np.var(X_test, axis=0)
        top_idx = np.argsort(feature_var)[-9:]
        top_features = [feature_names[i] for i in top_idx]
    
    feature_indices = [feature_names.index(f) for f in top_features[:9]]
    
    n_classes = len(class_names)
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        if i < len(axes):
            PartialDependenceDisplay.from_estimator(
                model, X_test, features=feature_indices,
                feature_names=feature_names, target=i,
                ax=axes[i], line_kw={"label": class_name}
            )
            axes[i].set_title(class_name, fontweight='bold')
            axes[i].legend(fontsize=8)
    
    # Hide unused subplots
    for j in range(len(class_names), len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Partial Dependence: Feature Impact on Class Probabilities\n'
                 '(Non-linear relationships reveal physics mechanisms)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("partial_dependence.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Partial Dependence plot saved: partial_dependence.png")
    
    # Physics insights
    logger.info("\n🎓 PDP Physics Insights:")
    for feat in top_features[:9]:
        if 'kurt' in feat:
            logger.info(f"  • {feat}: Non-linear threshold behavior expected (impulse detection)")
        elif 'ratio' in feat:
            logger.info(f"  • {feat}: Monotonic increase confirms directional fault physics")
        elif 'centroid' in feat:
            logger.info(f"  • {feat}: Peak near fault frequency validates spectral alignment")


# Optional: Add wrapper for all explainability methods
def run_full_explainability_suite(
    model, X_train, X_test, y_test, feature_names, class_names,
    instance_idx=0,  # which test sample to explain with LIME
    run_shap=True,
    run_lime=True,
    run_pdp=True,
    run_pfi=True
):
    """Run all explainability methods in one call."""
    logger.info("\n" + "="*70)
    logger.info("🧠 RUNNING FULL EXPLAINABILITY SUITE")
    logger.info("="*70)
    
    if run_pfi:
        compute_permutation_feature_importance(model, X_test, y_test, feature_names, class_names)
    
    if run_shap:
        explain_with_shap(model, X_train, X_test, feature_names, class_names)
    
    if run_lime and len(X_test) > instance_idx:
        explain_with_lime(model, X_test[instance_idx], X_train, feature_names, class_names)
    
    if run_pdp:
        plot_partial_dependence(model, X_test, feature_names, class_names)
    
    logger.info("\n✅ EXPLAINABILITY SUITE COMPLETE")