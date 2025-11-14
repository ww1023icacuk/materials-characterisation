import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import learning_curve


class Evaluator:

    def __init__(self, class_names=None):
        self.class_names = class_names

    # Basic metrics
    def compute_metrics(self, y_true, y_pred, average="binary", pos_label=1):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
        rec = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
        f1 = f1_score(y_true, y_pred, average=average, pos_label=pos_label)

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    def print_classification_report(self, y_true, y_pred):
        print(classification_report(y_true, y_pred))

        # Confusion matrix
    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        normalize=False,
        title="Confusion Matrix",
        save_path=None,
        save_metrics_path=None,
    ):
  
        cm = confusion_matrix(y_true, y_pred)

        # Normalize if requested
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        if save_metrics_path is not None:
            import pandas as pd
            tick_labels = (
                self.class_names if self.class_names is not None else np.unique(y_true)
            )
            df_cm = pd.DataFrame(
                cm,
                index=[f"Actual_{lbl}" for lbl in tick_labels],
                columns=[f"Pred_{lbl}" for lbl in tick_labels],
            )
            df_cm.to_csv(save_metrics_path)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        # Tick labels
        if self.class_names is not None:
            tick_labels = self.class_names
        else:
            tick_labels = np.unique(y_true)

        ax.set_xticks(np.arange(len(tick_labels)))
        ax.set_yticks(np.arange(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticklabels(tick_labels)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.tight_layout()

        # SAVE FIGURE YOU BITCH
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


    # Learning curve
    def plot_learning_curve(
        self,
        estimator,
        X,
        y,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy",
        n_jobs=None,
        title="Learning Curve",
    ):
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator=estimator,
            X=X,
            y=y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        plt.figure()
        plt.title(title)
        plt.xlabel("Training set size")
        plt.ylabel(scoring.capitalize())

        # Plot training scores
        plt.plot(train_sizes_abs, train_mean, marker="o", label="Training score")
        plt.fill_between(
            train_sizes_abs,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
        )

        # Plot validation scores
        plt.plot(train_sizes_abs, val_mean, marker="s", label="Validation score")
        plt.fill_between(
            train_sizes_abs,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.2,
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()