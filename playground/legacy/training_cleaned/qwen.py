# =============================================================================
# FILE: src/xai/run_xai_inference.py
# DESCRIPTION: Minimal XAI Pipeline - PFI + Integrated Gradients
#              ✅ FIXED: Forward pass evaluation order restored to 84.4% accuracy
#              ✅ FIXED: Feature names array alignment for PFI flattening
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import gc
import logging
import warnings
from typing import List, Optional

from captum.attr import IntegratedGradients
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler("xai_pfi_ig.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
plt.rcParams.update({
    'font.size': 11, 'figure.dpi': 200, 'axes.facecolor': 'white',
    'figure.facecolor': 'white', 'axes.linewidth': 1.2
})

COLORS = {'Axial': '#1f77b4', 'Radial': '#d62728', 'Tangential': '#2ca02c'}
AXIS_NAMES = ['Axial', 'Radial', 'Tangential']

FIG_DIR = Path("C:/DEV/Personal/Ai-Driven-Vibrations-motor/figures/xai")
RES_DIR = Path("C:/DEV/Personal/Ai-Driven-Vibrations-motor/src/results/xai")
MODEL_DIR = Path("C:/DEV/Personal/Ai-Driven-Vibrations-motor/src/training_cleaned/models/mafaulda_pytorch_order") 
DATA_DIR = Path("C:/DEV/Personal/Ai-Driven-Vibrations-motor/src/training_cleaned")

for d in [FIG_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 256
N_CHANNELS = 3
N_FEATURES = WINDOW_SIZE * N_CHANNELS

# ✅ FIXED: Corrected mapping order to match numpy's C-contiguous flattening
FEATURE_NAMES = [f"{AXIS_NAMES[c]}_{i}" for i in range(WINDOW_SIZE) for c in range(N_CHANNELS)]

# =============================================================================
# PHYSICS-INFORMED CNN MODEL
# =============================================================================
class PhysicsInformedCNN(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 6, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=31, padding=15),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=31, padding=15),
            nn.BatchNorm1d(64), nn.ReLU()
        )
        
        self.fuse = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=1),
            nn.BatchNorm1d(128), nn.ReLU()
        )
        
        if use_attention:
            self.se_fc1 = nn.Linear(128, 16)
            self.se_fc2 = nn.Linear(16, 128)
            self.radial_bias = nn.Parameter(torch.tensor(0.2))
            self.attn_conv = nn.Conv1d(128, 1, kernel_size=15, padding=7)
        
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor, rpm: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        x = self.fuse(x)
        
        if self.use_attention:
            se = self.global_pool(x).squeeze(-1)
            se = torch.sigmoid(self.se_fc2(torch.relu(self.se_fc1(se)))).unsqueeze(-1)
            x = x * se
            
            # RPM-based radial bias for misalignment detection
            if rpm is not None:
                misalignment_mask = (rpm > 1000) & (rpm < 3000)
                if misalignment_mask.any():
                    radial_multiplier = torch.ones_like(x)
                    radial_multiplier[:, 1, :] = 1.0 + self.radial_bias
                    x = x * radial_multiplier
            
            attn = torch.sigmoid(self.attn_conv(x))
            x = x * attn
        
        # ✅ CRITICAL FIX: Restored pooling *before* conv2 (Matches 84% accuracy)
        x = self.pool(x)
        x = self.conv2(x)
        
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)

# =============================================================================
# SKLEARN WRAPPER FOR PFI
# =============================================================================
class SKLearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model: nn.Module, device: torch.device, rpm: float = 1500.0):
        self.model = model
        self.device = device
        self.rpm = rpm
        self.classes_ = np.arange(6)
        self.model.eval()
    
    def fit(self, X, y=None): 
        return self
    
    def _prepare_input(self, X: np.ndarray) -> tuple:
        x_tensor = torch.FloatTensor(X.reshape(-1, WINDOW_SIZE, N_CHANNELS)).to(self.device)
        rpm_tensor = torch.ones(x_tensor.size(0), device=self.device) * self.rpm
        return x_tensor, rpm_tensor
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_tensor, rpm_tensor = self._prepare_input(X)
            return F.softmax(self.model(x_tensor, rpm_tensor), dim=1).cpu().numpy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

# =============================================================================
# XAI 1: PERMUTATION FEATURE IMPORTANCE
# =============================================================================
def compute_pfi(model: SKLearnWrapper, X_test: np.ndarray, y_test: np.ndarray,
                save_dir: Path, n_repeats: int = 10, n_samples: int = 500):
    logger.info("🔍 Computing Permutation Feature Importance (PFI)...")
    
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    logger.info(f"   Flattened data shape: {X_test_flat.shape}")
    
    if len(X_test_flat) > n_samples:
        idx = np.random.choice(len(X_test_flat), n_samples, replace=False)
        X_sample, y_sample = X_test_flat[idx], y_test[idx]
    else:
        X_sample, y_sample = X_test_flat, y_test
    
    logger.info(f"   Evaluating on {len(X_sample)} samples with {n_repeats} repeats...")
    
    try:
        result = permutation_importance(
            model, X_sample, y_sample, scoring='accuracy',
            n_repeats=n_repeats, random_state=42, n_jobs=1
        )
        
        pfi_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Importance_Mean': result.importances_mean,
            'Importance_Std': result.importances_std
        })
        
        axis_agg = []
        for axis in AXIS_NAMES:
            mask =[f.startswith(axis) for f in FEATURE_NAMES]
            axis_data = pfi_df[mask]
            axis_agg.append({
                'Axis': axis,
                'Mean_Importance': axis_data['Importance_Mean'].mean(),
                'Std_Importance': axis_data['Importance_Std'].mean(),
                'Max_Importance': axis_data['Importance_Mean'].max()
            })
        axis_df = pd.DataFrame(axis_agg)
        
        pfi_df.to_csv(save_dir / 'pfi_all_features.csv', index=False)
        axis_df.to_csv(save_dir / 'pfi_axis_aggregated.csv', index=False)
        
        # Plot 1: Top 20 features
        plt.figure(figsize=(14, 8))
        top_20 = pfi_df.nlargest(20, 'Importance_Mean')
        colors =[COLORS.get(name.split('_')[0], '#7f7f7f') for name in top_20['Feature']]
        plt.barh(range(len(top_20)), top_20['Importance_Mean'][::-1], 
                color=colors[::-1], alpha=0.85, edgecolor='black')
        plt.yticks(range(len(top_20)),[f"{f.split('_')[0]}[{f.split('_')[1]}]" 
                                       for f in top_20['Feature'][::-1]], fontsize=8)
        plt.xlabel('Mean Decrease in Accuracy')
        plt.title('🌍 PFI: Top 20 Most Important Features (Global)')
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(save_dir / 'pfi_top20_features.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Axis-level comparison
        plt.figure(figsize=(10, 6))
        plt.bar(axis_df['Axis'], axis_df['Mean_Importance'],
                color=[COLORS[ax] for ax in axis_df['Axis']], alpha=0.85,
                yerr=axis_df['Std_Importance'], capsize=5)
        plt.ylabel('Mean Feature Importance')
        plt.title('📊 PFI: Axis-Level Aggregated Importance')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(save_dir / 'pfi_axis_importance.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Timeline by axis
        plt.figure(figsize=(12, 6))
        for axis in AXIS_NAMES:
            mask =[f.startswith(axis) for f in FEATURE_NAMES]
            values = pfi_df[mask]['Importance_Mean'].values
            plt.plot(np.arange(WINDOW_SIZE), values, label=axis, color=COLORS[axis], linewidth=2)
        plt.xlabel('Time Sample Index')
        plt.ylabel('Permutation Importance')
        plt.title('📈 PFI: Feature Importance Timeline by Axis')
        plt.legend()
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(save_dir / 'pfi_timeline_by_axis.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ PFI completed - Results saved to {save_dir}")
        return axis_df
    except Exception as e:
        logger.error(f"❌ PFI failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# =============================================================================
# XAI 2: INTEGRATED GRADIENTS
# =============================================================================
def compute_integrated_gradients(model: nn.Module, X_sample: np.ndarray, 
                                y_sample: np.ndarray, rpm_sample: np.ndarray,
                                device: torch.device, class_names: List[str], 
                                save_dir: Path, n_samples: int = 50):
    logger.info("🔍 Computing Integrated Gradients (IG)...")
    model.eval()
    ig = IntegratedGradients(model)
    results =[]
    
    for cls_idx, cls_name in enumerate(class_names):
        mask = (y_sample == cls_idx)
        if np.sum(mask) == 0: 
            continue
        sample_indices = np.where(mask)[0]
        if len(sample_indices) > n_samples:
            sample_indices = np.random.choice(sample_indices, n_samples, replace=False)
        
        for idx in sample_indices:
            try:
                x_tensor = torch.FloatTensor(X_sample[idx][None, ...]).to(device)
                rpm_tensor = torch.FloatTensor([rpm_sample[idx]]).to(device)
                
                attributions, delta = ig.attribute(
                    x_tensor, 
                    target=cls_idx, 
                    baselines=torch.zeros_like(x_tensor),
                    n_steps=50, 
                    return_convergence_delta=True,
                    additional_forward_args=(rpm_tensor,)
                )
                
                attr_np = attributions[0].cpu().detach().numpy()
                axis_attr = {ax: np.abs(attr_np[:, i]).mean() for i, ax in enumerate(AXIS_NAMES)}
                
                results.append({
                    'sample_idx': int(idx), 
                    'true_class': class_names[y_sample[idx]],
                    'pred_class': cls_name,
                    'rpm': float(rpm_sample[idx]),
                    **{f'{ax.lower()}_importance': float(axis_attr[ax]) for ax in AXIS_NAMES}
                })
                pd.DataFrame(attr_np, columns=AXIS_NAMES).to_csv(
                    save_dir / f'ig_sample{idx}_{cls_name}.csv', index=False)
            except Exception as e:
                logger.warning(f"   IG failed for sample {idx}: {e}")
    
    if not results: 
        return None
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        plt.figure(figsize=(14, 8))
        melt_df = results_df.melt(id_vars=['pred_class'], 
                                 value_vars=[f'{ax.lower()}_importance' for ax in AXIS_NAMES],
                                 var_name='Axis', value_name='Attribution')
        melt_df['Axis'] = melt_df['Axis'].str.replace('_importance', '').str.title()
        sns.boxplot(data=melt_df, x='pred_class', y='Attribution', hue='Axis', palette=COLORS)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Integrated Gradient Attribution')
        plt.title('🎯 IG: Axis Attribution by Predicted Class')
        plt.tight_layout()
        plt.savefig(save_dir / 'ig_axis_distribution.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        results_df.to_csv(save_dir / 'ig_aggregated_results.csv', index=False)
        logger.info(f"✅ IG completed - {len(results)} samples processed")
    return results_df

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_minimal_xai(model_path: Optional[Path] = None,
                   data_path: Optional[Path] = None,
                   label_encoder_path: Optional[Path] = None):
    logger.info("="*70)
    logger.info("🚀 MINIMAL XAI: PFI + Integrated Gradients")
    logger.info("="*70)
    
    model_path = model_path or MODEL_DIR / "best_model.pt"
    data_path = data_path or DATA_DIR / "phase3_test_data.npz"
    label_encoder_path = label_encoder_path or MODEL_DIR / "label_encoder.pkl"
    
    for path, name in[(model_path,"Model"),(data_path,"Data"),(label_encoder_path,"Encoder")]:
        if not path.exists():
            logger.error(f"❌ {name} not found: {path}")
            return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"📍 Device: {device}")
    
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    class_names = label_encoder.classes_.tolist()
    logger.info(f"📋 Classes: {class_names}")
    
    model = PhysicsInformedCNN(num_classes=6, use_attention=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"✅ Model loaded from {model_path}")
    
    # Load data
    data = np.load(data_path)
    X_train, X_test, y_test = data['X_train'], data['X_test'], data['y_test']
    
    rpm_test = data.get('rpm_test', np.ones(len(X_test)) * 1500.0)
    rpm_train = data.get('rpm_train', np.ones(len(X_train)) * 1500.0)
    logger.info(f"📊 Data: Train={X_train.shape}, Test={X_test.shape}")
    logger.info(f"📊 RPM: Test={rpm_test.shape}, Range={rpm_test.min():.0f}-{rpm_test.max():.0f}")
    
    logger.info("\n📊 Evaluating model...")
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_test).to(device)
        rpm_tensor = torch.FloatTensor(rpm_test).to(device)
        outputs = model(x_tensor, rpm_tensor)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"✅ Test Accuracy: {acc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n📊 Confusion Matrix:\n{cm}")
    
    # Per-class accuracy
    for i, cls in enumerate(class_names):
        cls_acc = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
        logger.info(f"   {cls}: {cls_acc:.3f}")
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(RES_DIR / 'confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # PFI (Global)
    logger.info("\n🌍 STEP 1: Permutation Feature Importance")
    sk_model = SKLearnWrapper(model, device, rpm=1500.0)
    pfi_results = compute_pfi(sk_model, X_test, y_test, RES_DIR)
    
    # IG (Local)
    logger.info("\n🎯 STEP 2: Integrated Gradients")
    ig_results = compute_integrated_gradients(
        model, X_test, y_test, rpm_test, device, class_names, RES_DIR
    )
    
    gc.collect()
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
    
    logger.info("\n" + "="*70)
    logger.info("✅ COMPLETE")
    logger.info(f"📁 Results: {RES_DIR.absolute()}")
    logger.info("="*70)
    
    return {'pfi': pfi_results, 'ig': ig_results, 'accuracy': acc}

if __name__ == "__main__":
    run_minimal_xai()