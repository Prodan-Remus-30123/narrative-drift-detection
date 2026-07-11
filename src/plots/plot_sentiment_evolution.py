import matplotlib.pyplot as plt
from utils.period_sorting import (sort_period_key)
from utils.plot_saving import save_plot, DEFAULT_FIGSIZE

def plot_sentiment_evolution(sentiment_results, source, output_dir = None):

    periods = sorted(sentiment_results.keys(), key=sort_period_key)

    compound = [sentiment_results[p]["compound"] for p in periods]
    positive = [sentiment_results[p]["positive"] for p in periods]
    negative = [sentiment_results[p]["negative"] for p in periods]

    plt.figure(figsize=DEFAULT_FIGSIZE)

    plt.plot(
        periods,
        compound,
        marker="o",
        label="Compound"
    )

    plt.plot(
        periods,
        positive,
        marker="o",
        label="Positive"
    )

    plt.plot(
        periods,
        negative,
        marker="o",
        label="Negative"
    )

    plt.xticks(rotation=45)
    plt.ylabel("Sentiment")

    plt.title(f"{source} Sentiment Evolution")

    plt.legend()

    plt.tight_layout()

    if output_dir:
        save_plot(
            output_dir,
            plot_name=f"{source}_sentiment_evolution",
            source=source
        )
    else:
        plt.close()