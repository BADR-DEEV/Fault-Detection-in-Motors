import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from .config import DEVICE, EARLY_STOPPING_PATIENCE, MAX_EPOCHS
from .dataset import SpectralDataset
from .model import SpectralCNN
from .plotting import plot_learning_curves


def train_pytorch_model(X, y, groups, label_encoder, figures_dir, models_dir):
    num_classes = len(label_encoder.classes_)
    gkf = GroupKFold(n_splits=5)
    best_auc = 0.0
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]

        class_counts = np.bincount(y_train_fold)
        sample_weights = (1.0 / class_counts)[y_train_fold]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            SpectralDataset(X_train_fold, y_train_fold, augment=True),
            batch_size=64,
            sampler=sampler,
            num_workers=2 if torch.cuda.is_available() else 0,
        )
        val_loader = DataLoader(
            SpectralDataset(X[val_idx], y[val_idx], augment=False),
            batch_size=128,
            shuffle=False,
        )

        model = SpectralCNN(num_classes=num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

        best_fold_auc = 0.0
        patience_counter = 0
        fold_logs = []

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                correct += (torch.argmax(outputs, 1) == batch_y).sum().item()
                total += batch_y.size(0)

            train_acc = correct / total

            model.eval()
            val_loss = 0.0
            all_preds = []
            all_probs = []
            all_labels = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
                    all_probs.extend(torch.softmax(outputs, 1).cpu().numpy())
                    all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())

            val_acc = accuracy_score(all_labels, all_preds)
            try:
                val_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
            except Exception:
                val_auc = val_acc

            scheduler.step(val_loss / len(val_loader))
            fold_logs.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss / len(train_loader),
                    "train_acc": train_acc,
                    "val_loss": val_loss / len(val_loader),
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                }
            )

            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), models_dir / f"best_fold_{fold}.pt")
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

        plot_learning_curves(pd.DataFrame(fold_logs), fold, figures_dir)
        model.load_state_dict(torch.load(models_dir / f"best_fold_{fold}.pt", map_location=DEVICE))
        if best_fold_auc > best_auc:
            best_auc = best_fold_auc
            best_model = model
            torch.save(best_model.state_dict(), models_dir / "best_model.pt")

        del model
        gc.collect()

    return best_model
