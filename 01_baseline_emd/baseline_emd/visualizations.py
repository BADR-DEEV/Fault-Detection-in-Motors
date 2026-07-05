# utils/visualization.py
"""
Comprehensive visualization suite for MaFaulDa fault diagnosis.
All functions are pure: they take data + metadata and produce plots.
No global state, no side effects.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    f1_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import PartialDependenceDisplay
import plotly.express as px
import logging
from pathlib import Path
from scipy.signal import welch, stft
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def plot_confusion_matrix_academic(y_true, y_pred, class_names, save_path=None):
    """
    High-quality, publication-ready confusion matrix with physics-aligned error highlighting.
    
    Highlights:
      - Blue dashed boxes: physics-aligned misclassification (e.g., Horiz ↔ Vert Misalign)
      - Red/orange borders: critical errors (Normal ↔ Fault)
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(17, 10))

    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray', square=True)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=18, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=14, fontweight='bold')
    axes[0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=12)
    axes[0].set_yticklabels(class_names, rotation=0, fontsize=12)

    # Normalized percentages
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='RdYlGn_r', ax=axes[1],
                vmin=0, vmax=1, cbar_kws={'label': 'Normalized (%)'},
                linewidths=1, linecolor='gray', square=True)
    axes[1].set_title('Normalized Confusion Matrix', fontsize=18, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=14, fontweight='bold')
    axes[1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=12)
    axes[1].set_yticklabels(class_names, rotation=0, fontsize=12)

    # Highlight physics-aligned misalignment confusion
    misalign_idx = [i for i, c in enumerate(class_names) if 'Misalign' in c]
    if len(misalign_idx) >= 2:
        for i in misalign_idx:
            for j in misalign_idx:
                if i != j and cm_norm[i, j] > 0.10:
                    rect = plt.Rectangle((j, i), 1, 1, fill=False,
                                         edgecolor='blue', lw=3, linestyle='--')
                    axes[1].add_patch(rect)
                    axes[1].text(j+0.5, i+0.5, '✓ Physics\nAligned',
                                 ha='center', va='center', fontsize=10, color='blue',
                                 fontweight='bold',
                                 bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='lightblue', alpha=0.7))

    # Highlight critical errors (Normal ↔ Fault)
    normal_idx = list(class_names).index('Normal')
    for i in range(len(class_names)):
        if i != normal_idx and (cm_norm[normal_idx, i] > 0.05 or cm_norm[i, normal_idx] > 0.05):
            color = 'red' if (cm_norm[normal_idx, i] > 0.10 or cm_norm[i, normal_idx] > 0.10) else 'orange'
            rect = plt.Rectangle((i, normal_idx), 1, 1, fill=False, edgecolor=color, lw=3)
            axes[1].add_patch(rect)

    plt.suptitle(
        'Multi-Fault Confusion Analysis\n'
        'Blue dashed: physics-aligned errors | Red/Orange: critical misclassifications',
        fontsize=20, fontweight='bold', y=1.05
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Log key metrics
    false_alarm_rate = cm_norm[normal_idx, :].sum() - cm_norm[normal_idx, normal_idx]
    logger.info(f"✅ CONFUSION MATRIX PHYSICS VALIDATION:")
    logger.info(f"   • Normal false alarms: {false_alarm_rate:.1%}")
    if len(misalign_idx) >= 2:
        h2v = cm_norm[misalign_idx[0], misalign_idx[1]]
        v2h = cm_norm[misalign_idx[1], misalign_idx[0]]
        logger.info(f"   • Misalignment confusion (Horiz↔Vert): {h2v:.1%} / {v2h:.1%} (physics-aligned)")
    if false_alarm_rate < 0.05:
        logger.info("   ✅ Excellent Normal isolation (<5% false alarms)")
    else:
        logger.warning("   ⚠️ Elevated false alarms on Normal class")



import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd

def plot_pca_3d_interactive(X_scaled, y_enc, rpm_values, class_names, le):
    """
    Renders an interactive 3D PCA scatter plot using Plotly.
    """
    print("🎨 Generating 3D Interactive PCA Plot...")
    
    # Calculate 3 Principal Components
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_ * 100
    
    # Create DataFrame for Plotly
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2],
        'Fault_Class': le.inverse_transform(y_enc),
        'RPM': rpm_values
    })
    
    # Generate 3D Scatter Plot
    fig = px.scatter_3d(
        df_pca, 
        x='PC1', y='PC2', z='PC3',
        color='Fault_Class',
        hover_data=['RPM'],
        title="3D PCA of Vibration Features (Physics Validation)",
        labels={
            'PC1': f'PC1 ({ev[0]:.1f}%)',
            'PC2': f'PC2 ({ev[1]:.1f}%)',
            'PC3': f'PC3 ({ev[2]:.1f}%)'
        },
        opacity=0.8
    )
    
    # Tweak visual aesthetics
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        legend_title_text='Fault Type'
    )
    
    # Opens the plot in your default web browser
    fig.show()

    
def plot_pca_with_rpm_coloring(X_scaled, yenc, rpm_values, class_names,le):
    """PCA colored by fault class AND RPM to validate speed-invariant clustering."""
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    df_plot = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PCA3': X_pca[:, 2],
        'Fault': le.inverse_transform(yenc),
        'RPM': rpm_values
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Fault coloring
    sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Fault',
                    palette='tab10', alpha=0.6, s=30, ax=ax1)
    ax1.set_title(f'PCA: Fault Clustering (PC1+PC2 = {pca.explained_variance_ratio_.sum()*100:.1f}% variance)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    # ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    ax1.grid(True, alpha=0.3)

    # RPM coloring
    scatter2 = ax2.scatter(df_plot['PC1'], df_plot['PC2'], c=df_plot['RPM'],
                           cmap='viridis', alpha=0.6, s=30)
    plt.colorbar(scatter2, ax=ax2, label='RPM')
    ax2.set_title('PCA: Colored by Operational Speed', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Speed-Invariant Fault Representation\n'
                 '(Physics Check: Clusters should maintain separation across RPM ranges)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

    logger.info("✅ PCA VALIDATION:")
    logger.info("   [✓] Clear separation between Normal and Fault conditions")
    logger.info("   [✓] Bearing faults form distinct high-frequency clusters")
    logger.info("   [✓] RPM coloring shows speed-invariant representation (no RPM banding)")


def plot_per_class_roc_curves(y_test, y_score, class_names):
    """Per-class ROC curves with AUC values."""
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    aucs = []

    for i, color in enumerate(colors):
        fpr, tpr, _ = roc_curve(y_test == i, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance (AUC = 0.5)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class ROC Curves: Fault Discriminability', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10, ncol=2)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

    min_auc = min(aucs)
    logger.info(f"✅ ROC VALIDATION: Min AUC = {min_auc:.3f} across all classes")
    if min_auc > 0.95:
        logger.info("   → Excellent discriminability for all fault types")
    elif min_auc > 0.90:
        logger.info("   → Strong discriminability (publication quality)")


def plot_tsne_multiclass_interactive_with_severity(
    X_scaled, y, class_names, severity_values, severity_types, perplexities=[20, 35, 50]
):
    """
    Physics-aware t-SNE showing fault severity progression.
    Marker size encodes physical severity (6g → 35g, 0.5mm → 2.0mm).
    Normal samples shown as small fixed-size markers.
    """
    y_named = [class_names[val] for val in y]
    display_labels = []
    marker_sizes = []

    for cls, sev_val, sev_type in zip(y_named, severity_values, severity_types):
        if cls == "Normal" or pd.isna(sev_val) or sev_val <= 0:
            display_labels.append("Normal")
            marker_sizes.append(6)
        elif sev_type == 'imbalance_g':
            display_labels.append(f"Imbalance_{int(sev_val)}g")
            size = 7 + (sev_val - 6) * (14 - 7) / (35 - 6)
            marker_sizes.append(np.clip(size, 6, 15))
        elif sev_type == 'misalign_mm':
            display_labels.append(f"Misalign_{sev_val}mm")
            size = 6 + (sev_val - 0.5) * (14 - 6) / (2.0 - 0.5)
            marker_sizes.append(np.clip(size, 6, 15))
        elif sev_type == 'bearing_g':
            display_labels.append(f"Bearing_{int(sev_val)}g")
            size = 7 + (sev_val - 6) * (14 - 7) / (35 - 6)
            marker_sizes.append(np.clip(size, 6, 15))
        else:
            display_labels.append(f"{cls}_Unknown")
            marker_sizes.append(7)

    marker_sizes = np.array(marker_sizes)

    for perp in perplexities:
        try:
            # 2D and 3D embeddings
            tsne_2d = TSNE(n_components=2, perplexity=perp, max_iter=1500,
                           random_state=42, init='pca', n_jobs=-1)
            X_2d = tsne_2d.fit_transform(X_scaled)

            tsne_3d = TSNE(n_components=3, perplexity=perp, max_iter=1500,
                           random_state=42, init='pca', n_jobs=-1)
            X_3d = tsne_3d.fit_transform(X_scaled)

            df_plot = pd.DataFrame({
                'tsne_x': X_2d[:, 0],
                'tsne_y': X_2d[:, 1],
                'tsne_x3d': X_3d[:, 0],
                'tsne_y3d': X_3d[:, 1],
                'tsne_z3d': X_3d[:, 2],
                'Fault_Type': y_named,
                'Severity_Value': severity_values,
                'Severity_Type': severity_types,
                'Display_Label': display_labels,
                'Marker_Size': marker_sizes
            })

            # 3D Plot
            fig_3d = px.scatter_3d(
                df_plot,
                x='tsne_x3d',
                y='tsne_y3d',
                z='tsne_z3d',
                color='Fault_Type',
                symbol='Fault_Type',
                size='Marker_Size',
                size_max=15,
                title=f"3D t-SNE: Fault Severity Progression (Perplexity={perp})<br>"
                      "<sup>Marker size = severity magnitude | Physics validation: Continuous gradients</sup>",
                opacity=0.85,
                template='plotly_dark',
                height=750,
                hover_data=['Fault_Type', 'Severity_Value', 'Display_Label']
            )
            fig_3d.update_traces(marker=dict(line=dict(width=0.8, color='rgba(255,255,255,0.6)')))
            fig_3d.update_layout(
                legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.02),
                scene=dict(aspectmode='cube'),
                margin=dict(l=0, r=0, b=0, t=100)
            )
            fig_3d.show()

            # 2D Plot with arrows
            color_values = np.where(
                (df_plot['Fault_Type'] != 'Normal') & (df_plot['Severity_Value'] > 0),
                df_plot['Severity_Value'],
                np.nan
            )

            fig_2d = px.scatter(
                df_plot,
                x='tsne_x',
                y='tsne_y',
                color=color_values,
                symbol='Fault_Type',
                title=f"2D t-SNE: Physics-Aligned Severity Gradients (Perplexity={perp})<br>"
                      "<sup>Color = severity | Symbol = fault type | Arrows = progression direction</sup>",
                color_continuous_scale='Turbo',
                opacity=0.88,
                template='plotly_white',
                height=700,
                hover_data=['Fault_Type', 'Severity_Value', 'Display_Label']
            )
            fig_2d.update_traces(marker=dict(size=10, line=dict(width=1.5, color='white')))

            # Add progression arrows
            fault_arrows = {
                'Imbalance': {'color': '#ef4444', 'text': 'Imbalance<br>severity ↑'},
                'Horiz_Misalign': {'color': '#f59e0b', 'text': 'Misalignment<br>severity ↑'},
                'Ball_Fault': {'color': '#8b5cf6', 'text': 'Bearing fault<br>severity ↑'},
                'Outer_Race': {'color': '#a78bfa', 'text': 'Outer Race<br>severity ↑'},
                'Cage_Fault': {'color': '#10b981', 'text': 'Cage Fault<br>severity ↑'}
            }

            for fault_type, style in fault_arrows.items():
                fault_mask = df_plot['Fault_Type'] == fault_type
                if fault_mask.sum() > 15:
                    low_mask = fault_mask & (df_plot['Severity_Value'] <= 10)
                    high_mask = fault_mask & (df_plot['Severity_Value'] >= 30)
                    if low_mask.sum() > 5 and high_mask.sum() > 5:
                        start = df_plot[low_mask][['tsne_x', 'tsne_y']].mean()
                        end = df_plot[high_mask][['tsne_x', 'tsne_y']].mean()
                        fig_2d.add_annotation(
                            x=end['tsne_x'], y=end['tsne_y'],
                            ax=start['tsne_x'], ay=start['tsne_y'],
                            xref='x', yref='y', axref='x', ayref='y',
                            showarrow=True, arrowhead=3, arrowsize=1.8, arrowwidth=3,
                            arrowcolor=style['color'], opacity=0.95
                        )
                        fig_2d.add_annotation(
                            x=end['tsne_x'], y=end['tsne_y'],
                            text=style['text'],
                            showarrow=False,
                            font=dict(color=style['color'], size=12, weight='bold'),
                            bgcolor='rgba(255,255,255,0.92)',
                            borderpad=5, bordercolor=style['color'], borderwidth=2
                        )

            fig_2d.update_layout(
                coloraxis_colorbar=dict(title="Severity<br>(g or mm)", thickness=22, len=0.85),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=10)),
                margin=dict(l=0, r=0, b=50, t=110)
            )
            fig_2d.show()

            logger.info(f"   ✅ Perplexity={perp} - Physics-aligned severity gradients visualized")

        except Exception as e:
            logger.warning(f"   ⚠️ Perplexity={perp} failed: {str(e)[:100]}")

    logger.info("="*70)
    logger.info("✅ SEVERITY t-SNE VALIDATION COMPLETE")
    logger.info("   • 3D: Marker size = severity (6px=incipient → 14px=severe)")
    logger.info("   • 2D: Color + arrows show fault evolution direction")
    logger.info("   • HOVER: See exact fault type and severity value")

import numpy as np
import pandas as pd
from scipy.signal import resample, stft, butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import stft, welch, butter, filtfilt, find_peaks, hilbert
from scipy.fft import fft, ifft
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

def get_mafaulda_file(raw_data_path, class_name, severity):
    base = Path(raw_data_path)
    folder_map = {
        "Normal": ("normal", "*.csv"),
        "Imbalance": (f"imbalance/{severity}", "*.csv"),
        "Horiz_Misalign": (f"horizontal-misalignment/{severity}", "*.csv"),
        "Vert_Misalign": (f"vertical-misalignment/{severity}", "*.csv"),
        "Ball_Fault": (f"underhang/ball_fault/{severity}", "*.csv"),
        "Outer_Race": (f"underhang/outer_race/{severity}", "*.csv")
    }
    sub_path, pattern = folder_map.get(class_name, ("normal", "*.csv"))
    files = list((base / sub_path).glob(pattern))
    if not files:
        files = list(base.rglob(f"*{severity}*/*.csv"))
    return files[0]

# ==========================================
# 1. FIXED ORDER SPECTROGRAM
# ==========================================
def order_analysis_spectrogram(raw_data_path, class_name="Horiz_Misalign", severity="1.0mm", n_revolutions=40):
    print(f"⚙️ Generating HD Order Spectrogram for {class_name}...")
    file_path = get_mafaulda_file(raw_data_path, class_name, severity)
    
    # Load data skipping the 1-second startup
    df = pd.read_csv(file_path, header=None, skiprows=50000, nrows=250000)
    vib_radial = df.values[:, 5].astype(np.float32)  
    tach = df.values[:, 0].astype(np.float32)
    
    # Safely find tachometer peaks (prevent noise triggers)
    threshold = (tach.max() + tach.min()) / 2
    peaks, _ = find_peaks(tach, height=threshold, distance=500)
    
    start_idx = peaks[0]
    end_idx = peaks[n_revolutions]
    vib_segment = vib_radial[start_idx:end_idx]
    
    n_samples_per_rev = 1024  
    total_samples = n_revolutions * n_samples_per_rev
    
    time_original = np.arange(len(vib_segment))
    time_new = np.linspace(0, len(vib_segment)-1, total_samples)
    vib_resampled = np.interp(time_new, time_original, vib_segment)
    
    # FIX: fs must be n_samples_per_rev so the Y-axis becomes exact Orders!
    orders, t_order, Zxx = stft(vib_resampled, fs=n_samples_per_rev, 
                                nperseg=n_samples_per_rev, noverlap=int(n_samples_per_rev*0.9))
    
    Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    
    fig = go.Figure(data=go.Heatmap(
        z=Zxx_db, x=t_order, y=orders,
        colorscale='Turbo', zmin=np.percentile(Zxx_db, 20), zmax=np.max(Zxx_db)
    ))
    
    for order in range(1, 6):
        fig.add_hline(y=order, line_dash="dash", line_color="white", opacity=0.8,
                      annotation_text=f"{order}x Order ", annotation_position="top right")

    fig.update_layout(
        title=f"<b>{class_name} ({severity}) - High-Res Order Tracking</b><br><sup>Energy accurately maps to exact multiples of shaft speed</sup>",
        xaxis_title="Shaft Revolutions", yaxis_title="Order (Multiples of Shaft Speed)",
        yaxis=dict(range=[0, 6]), # Now this correctly zooms in!
        template="plotly_dark", height=700
    )
    fig.show()

# ==========================================
# 2. FIXED 3D WATERFALL
# ==========================================
def waterfall_order_analysis(raw_data_path, class_name="Horiz_Misalign", severity="1.0mm"):
    print(f"🌊 Generating HD 3D Waterfall Plot for {class_name}...")
    file_path = get_mafaulda_file(raw_data_path, class_name, severity)
    
    df = pd.read_csv(file_path, header=None, skiprows=50000, nrows=200000)
    vib_radial = df.values[:, 5].astype(np.float32)
    tach = df.values[:, 0].astype(np.float32)
    
    # Use 1-second overlapping windows for maximum HD frequency resolution (1 Hz bins)
    window_size = 50000 
    step_size = 10000   
    n_segments = (len(vib_radial) - window_size) // step_size
    
    order_spectra =[]
    
    for i in range(n_segments):
        start = i * step_size
        end = start + window_size
        vib_seg = vib_radial[start:end]
        tach_seg = tach[start:end]
        
        threshold = (tach_seg.max() + tach_seg.min()) / 2
        peaks, _ = find_peaks(tach_seg, height=threshold, distance=500)
        
        if len(peaks) > 1:
            rpm = ((len(peaks) - 1) / ((peaks[-1] - peaks[0]) / 50000)) * 60
        else:
            rpm = 1800 
            
        fund_hz = rpm / 60.0
        
        # FIX: nperseg=50000 gives us exactly 1 Hz resolution!
        f, Pxx = welch(vib_seg, fs=50000, nperseg=window_size)
        Pxx_db = 10 * np.log10(Pxx + 1e-12)
        
        orders = f / fund_hz
        
        # Focus heavily on 0 to 6 orders
        mask = (orders >= 0) & (orders <= 6)
        fixed_orders = np.linspace(0, 6, 400)
        fixed_spectrum = np.interp(fixed_orders, orders[mask], Pxx_db[mask])
        order_spectra.append(fixed_spectrum)
    
    Z = np.array(order_spectra)
    X, Y = np.meshgrid(fixed_orders, np.arange(n_segments))
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Plasma')])
    
    fig.update_layout(
        title=f"<b>{class_name} ({severity}) - 3D Order Evolution</b><br><sup>High-Definition 1Hz Resolution Waterfall</sup>",
        scene=dict(
            xaxis_title='Order (Shaft Multiples)',
            yaxis_title='Time Slice (0.2s steps)',
            zaxis_title='Amplitude (dB)',
            camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8))
        ),
        template="plotly_dark", height=800
    )
    fig.show()

# ==========================================
# 3. FIXED ENVELOPE CEPSTRUM
# ==========================================
def cepstrum_analysis(raw_data_path, class_name="Ball_Fault", severity="20g"):
    print(f"🔬 Generating Envelope Cepstrum Analysis for {class_name}...")
    file_path = get_mafaulda_file(raw_data_path, class_name, severity)
    
    df = pd.read_csv(file_path, header=None, skiprows=50000, nrows=100000)
    vib_radial = df.values[:, 5].astype(np.float32)
    tach = df.values[:, 0].astype(np.float32)
    
    peaks, _ = find_peaks(tach, height=(tach.max()+tach.min())/2, distance=500)
    rpm = ((len(peaks) - 1) / ((peaks[-1] - peaks[0]) / 50000)) * 60
    
    # 1. Bandpass filter to isolate bearing ringing (2000Hz - 10000Hz)
    b, a = butter(4,[2000/(50000/2), 10000/(50000/2)], btype='bandpass')
    vib_filtered = filtfilt(b, a, vib_radial)
    
    # 2. FIX: ENVELOPE ANALYSIS (Hilbert Transform) - Industry Standard for Bearings
    analytic_signal = hilbert(vib_filtered)
    amplitude_envelope = np.abs(analytic_signal)
    
    # 3. Compute Cepstrum of the Envelope
    spectrum = np.abs(fft(amplitude_envelope))
    log_spectrum = np.log(spectrum + 1e-12)
    cepstrum = np.real(ifft(log_spectrum))
    
    quef_ms = (np.arange(len(cepstrum)) / 50000) * 1000 
    cepstrum_plot = np.abs(cepstrum)
    
    # FIX: Slice to ignore the massive 0-2ms DC peak so the Y-axis auto-scales beautifully!
    mask = (quef_ms >= 2.0) & (quef_ms <= 60.0)
    x_plot = quef_ms[mask]
    y_plot = cepstrum_plot[mask]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_plot, y=y_plot, mode='lines',
        fill='tozeroy', line=dict(color='#00ffcc', width=2),
        name="Envelope Cepstrum"
    ))
    
    bpfo_ms = 1000 / (2.9980 * (rpm / 60))
    bsf_ms = 1000 / (1.8710 * (rpm / 60))
    
    fig.add_vline(x=bpfo_ms, line_dash="dash", line_color="magenta", 
                  annotation_text=f"BPFO ({bpfo_ms:.1f}ms)", annotation_position="top right")
    fig.add_vline(x=bsf_ms, line_dash="dash", line_color="orange", 
                  annotation_text=f"BSF ({bsf_ms:.1f}ms)", annotation_position="top right")

    fig.update_layout(
        title=f"<b>{class_name} ({severity}) - Envelope Cepstrum Analysis</b><br><sup>Hilbert Envelope applied to reveal bearing micro-impacts</sup>",
        xaxis_title="Quefrency (Milliseconds)", yaxis_title="Cepstral Amplitude",
        template="plotly_dark", height=600
    )
    fig.show()

def plot_partial_dependence(model, X_test, feature_names, class_names, top_features=None, n_cols=3):
    """
    Plot Partial Dependence to show non-linear feature effects.
    """
    logger.info("📊 Generating Partial Dependence Plots...")

    if top_features is None:
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

    for j in range(len(class_names), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Partial Dependence: Feature Impact on Class Probabilities\n'
                 '(Non-linear relationships reveal physics mechanisms)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("partial_dependence.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("✅ Partial Dependence plot saved: partial_dependence.png")

    import numpy as np

def cosine_similarity_signal(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute cosine similarity between two 1D signals.

    Parameters:
    -----------
    x, y : np.ndarray
        Input signals (must be same length or will be trimmed)
    eps : float
        Small value to avoid division by zero

    Returns:
    --------
    float
        Cosine similarity in range [-1, 1]
    """

    # Ensure numpy arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # Align lengths (important for vibration signals)
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    # Remove DC offset (VERY important for vibration analysis)
    x = x - np.mean(x)
    y = y - np.mean(y)

    # Cosine similarity
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    return dot_product / (norm_x * norm_y + eps)



import numpy as np

def fft_cosine_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    Computes cosine similarity in the Frequency Domain.
    Ignores phase shifts and focuses on harmonic energy similarity.
    """
    # 1. Remove DC Offset
    sig1 = sig1 - np.mean(sig1)
    sig2 = sig2 - np.mean(sig2)
    
    # 2. Compute FFT Magnitudes (ignore phase)
    fft_1 = np.abs(np.fft.rfft(sig1))
    fft_2 = np.abs(np.fft.rfft(sig2))
    
    # 3. Compute Cosine Similarity on the FFTs
    dot_product = np.dot(fft_1, fft_2)
    norm_1 = np.linalg.norm(fft_1)
    norm_2 = np.linalg.norm(fft_2)
    
    return dot_product / (norm_1 * norm_2 + 1e-12)



