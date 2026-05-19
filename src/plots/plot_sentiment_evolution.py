import matplotlib.pyplot as plt


def plot_sentiment_evolution(sentiment_results, source):

    periods = list(sentiment_results.keys())

    compound = [sentiment_results[p]["compound"] for p in periods]
    positive = [sentiment_results[p]["positive"] for p in periods]
    negative = [sentiment_results[p]["negative"] for p in periods]

    plt.figure(figsize=(12, 6))

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

    plt.show()