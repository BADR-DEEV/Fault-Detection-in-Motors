import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_learning_curves(history_df, fold, output_dir):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="blue", linewidth=2)
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="red", linewidth=2)
    plt.title(f"Fold {fold}: Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Train Acc", color="blue", linewidth=2)
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Val Acc", color="red", linewidth=2)
    plt.title(f"Fold {fold}: Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / f"learning_curves_fold_{fold}.png", dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Test Set Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()
