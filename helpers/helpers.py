import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import numpy as np

colors = {'Graduate': '#A8E6CF', 'Enrolled': '#FCFC99', 'Dropout': '#FF8B94'}

def plot_precision_recall_vs_threshold_multi(model_results, target_recall=0.95, ncols=2):
    """
    Plot precision and recall vs. threshold curves for multiple models,
    and highlight the threshold that achieves the specified recall.

    Parameters:
    - model_results: dict of {model_name: (y_true, y_scores)}
    - target_recall: desired recall level (e.g., 0.95)
    - ncols: number of subplot columns
    """
    n_models = len(model_results)
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    res_threshold = {}

    for idx, (model_name, (y_true, y_scores)) in enumerate(model_results.items()):
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

        # Find the threshold where recall is just above or equal to target
        recall_diffs = np.abs(recalls - target_recall)
        best_index = recall_diffs.argmin()
        best_threshold = thresholds[max(best_index - 1, 0)]  # thresholds has len n-1

        ax = axes[idx]
        ax.plot(thresholds, precisions[:-1], "-", color=colors["Graduate"], label="Precision", alpha=1, linewidth=2.0)
        ax.plot(thresholds, recalls[:-1], "-", color=colors["Dropout"], label="Recall", alpha=1, linewidth=2.0)
        ax.axvline(x=best_threshold, color='gray', linestyle='--',
                   label=f'Thresh @ Recall={recalls[best_index]:.2f}')

        ax.set_title(model_name)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.grid(True)
        ax.legend(loc="lower right")
        res_threshold[model_name] = best_threshold

    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    plt.show()
    return res_threshold