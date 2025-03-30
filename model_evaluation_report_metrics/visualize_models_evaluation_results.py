import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def visualize_eval_results(df_eval, metrics, title):
    # Sort models by Overall Accuracy (OA) for better readability
    # df_eval = df_eval.sort_values(by="OA", ascending=False)

    # Set a modern color palette
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2", len(metrics))

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df_eval.set_index("Combination of bands")[metrics].plot(kind="bar", ax=ax, color=palette, width=0.8)

    # Add titles and labels
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xlabel("Combination of Bands")
    ax.set_xticklabels(df_eval["Combination of bands"], rotation=30, ha="right")

    # Add value labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.4f}", (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=10)

    # Move the legend outside to avoid clutter
    ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Show plot
    plt.tight_layout()
    plt.show()
