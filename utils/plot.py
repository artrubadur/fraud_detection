import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc


# Analysis
def num_features(features, df):

    for feature in features:
        values = df[feature]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(data=values, kde=True)
        plt.title(f"Distribution of {feature}")
        plt.xlabel("")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        sns.boxplot(data=values, orient="h")
        plt.title(f"Boxplot of {feature}")
        plt.xlabel("")
        plt.ylabel("")

        plt.tight_layout()
        plt.show()


def bi_num_features(features, df, columns=4):
    rows = len(features) // columns + 1

    _, axes = plt.subplots(rows, columns, figsize=(12, rows * 4))

    for i, ax in enumerate(axes.flat):
        if i >= len(features):
            ax.set_visible(False)
            continue

        feature = features[i]
        corr = df[[feature, "amt"]].corr().iloc[0, 1]

        sns.scatterplot(ax=ax, x=df[feature], y=df["amt"], label=f"{corr:.3f}")
        ax.set_title(f"{feature} vs. Amount")
        ax.set_xlabel(feature)
        ax.set_ylabel("Amount")
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def bi_cat_features(features, df):
    for feature in features:

        # Most popular
        top_categories = df[feature].value_counts().head(20).index
        filtered_data = df[df[feature].isin(top_categories)]
        top_means = (
            filtered_data.groupby(feature, observed=True)["is_fraud"]
            .mean()
            .reset_index()
        )
        top_means = top_means.sort_values("is_fraud", ascending=False)
        top_means[feature] = top_means[feature].astype(str)

        # Highest risk
        means = df.groupby(feature, observed=True)["is_fraud"].mean().reset_index()
        means = means.sort_values("is_fraud", ascending=False)[:20]
        means[feature] = means[feature].astype(str)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.barplot(data=top_means, x=feature, y="is_fraud")
        plt.title(f"Most popular {feature}")
        plt.xlabel(feature)
        plt.xticks(rotation=90)
        plt.ylabel("Fraud rate")

        plt.subplot(1, 2, 2)
        sns.barplot(data=means, x=feature, y="is_fraud")
        plt.title(f"Highest risk {feature}")
        plt.xlabel(feature)
        plt.xticks(rotation=90)
        plt.ylabel("Fraud rate")

        plt.tight_layout()
        plt.show()


def correlation_matrix(correlations):
    plt.figure(figsize=(16, 12))

    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        xticklabels=True,
        yticklabels=True,
        annot_kws={"size": 8},
    )
    plt.xticks(
        ticks=np.arange(len(correlations.columns)) + 0.5,
        labels=list(correlations.columns),
        rotation=45,
        ha="right",
    )
    plt.yticks(
        ticks=np.arange(len(correlations.index)) + 0.5,
        labels=list(correlations.index),
        rotation=0,
    )

    plt.tight_layout()
    plt.show()


# Work
def threshold_curve(
    threshold_value, confusion_thresholds, fns, fps, cost_thresholds, costs
):
    idx = np.argmin(np.abs(cost_thresholds - threshold_value))
    cost = costs[idx]

    idx = np.argmin(np.abs(confusion_thresholds - threshold_value))
    fn = fns[idx]

    idx = np.argmin(np.abs(confusion_thresholds - threshold_value))
    fp = fps[idx]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.lineplot(x=confusion_thresholds, y=fns, label=f"FN {fn}")
    sns.lineplot(x=confusion_thresholds, y=fps, label=f"FP {fp}")
    plt.axvline(
        x=threshold_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold {threshold_value:.2f} ",
    )

    plt.xlabel("Threshold")
    plt.ylabel("Items")
    plt.title("Threshold-Confusion Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.lineplot(x=cost_thresholds, y=costs, label="Costs")
    plt.axvline(
        x=threshold_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold {threshold_value:.2f} (C={cost:.2f})",
    )

    plt.xlabel("Threshold")
    plt.ylabel("Costs")
    plt.title("Threshold-Costs Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()


def roc_curve(threshold_value, thresholds, fpr, tpr):
    idx = np.argmin(np.abs(thresholds - threshold_value))
    fpr_threshold, tpr_threshold = fpr[idx], tpr[idx]
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))

    sns.lineplot(x=fpr, y=tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    sns.scatterplot(
        x=[fpr_threshold],
        y=[tpr_threshold],
        color="red",
        s=50,
        label=f"Threshold {threshold_value:.2f}",
        zorder=2,
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()


def precision_recall_curve(threshold_value, thresholds, precision, recall, ap):
    idx = np.argmin(np.abs(thresholds - threshold_value))
    precision_threshold, recall_threshold = precision[idx], recall[idx]

    plt.figure(figsize=(6, 6))

    sns.lineplot(x=recall, y=precision, label=f"PR curve (AP = {ap:.2f})")
    sns.scatterplot(
        x=[recall_threshold],
        y=[precision_threshold],
        color="red",
        s=50,
        label=f"Threshold {threshold_value:.2f} (P={precision_threshold:.2f}, R={recall_threshold:.2f})",
        zorder=2,
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()


def confusion_matrix(matrix):
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        xticklabels=True,
        yticklabels=True,
    )
    plt.xticks(
        ticks=[0.5, 1.5],
        labels=["Negative", "Positive"],
    )
    plt.yticks(
        ticks=[0.5, 1.5],
        labels=["Negative", "Positive"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.show()


def feature_importance(features, importances, top=15):
    plt.figure(figsize=(top * 0.8, 6))

    plt.subplot(1, 2, 1)
    sns.barplot(x=features[:top], y=importances[:top])
    plt.ylabel("Feature Importance")
    plt.xlabel("")
    plt.title("Top 15 Most Important Features")
    plt.xticks(
        ticks=np.arange(top) + 0.5,
        labels=list(features[:top]),
        rotation=45,
        ha="right",
    )

    plt.subplot(1, 2, 2)
    sns.barplot(x=features[-top:], y=importances[-top:])
    plt.ylabel("Feature Importance")
    plt.xlabel("")
    plt.title("Top 15 Less Important Features")
    plt.xticks(
        ticks=np.arange(top) + 0.5,
        labels=list(features[-top:]),
        rotation=45,
        ha="right",
    )

    plt.show()
