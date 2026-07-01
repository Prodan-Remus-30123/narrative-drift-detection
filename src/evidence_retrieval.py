import numpy as np
import pandas as pd


def _cosine_similarity(vec_a, vec_b):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _period_from_date(date_value):
    """
    Converts article date to bimonthly period label:
    2020-01, 2020-02 -> 2020-01_02
    2020-03, 2020-04 -> 2020-03_04
    """

    date = pd.to_datetime(date_value, errors="coerce")

    if pd.isna(date):
        return None

    year = date.year
    month = date.month

    if month % 2 == 1:
        start_month = month
        end_month = month + 1
    else:
        start_month = month - 1
        end_month = month

    return f"{year}-{start_month:02d}_{end_month:02d}"


def _split_transition(transition_label):
    if "->" not in transition_label:
        return None, None

    before, after = transition_label.split("->", 1)

    return before.strip(), after.strip()


def _get_period_centroid(source_result, period):
    trajectory = source_result.get(
        "semantic_embedding_trajectory",
        {}
    )

    labels = trajectory.get("labels", [])
    vectors = trajectory.get("vectors", [])

    for label, vector in zip(labels, vectors):
        if str(label) == str(period):
            return vector

    return None


def _article_record(row, similarity):
    text = str(row.get("text", ""))

    return {
        "title": row.get("title", ""),
        "url": row.get("url", ""),
        "date": str(row.get("date", "")),
        "source": row.get("source", ""),
        "similarity_to_period_centroid": similarity,
        "excerpt": text[:700].replace("\n", " ")
    }


def retrieve_representative_articles_for_period(
    df,
    source,
    period,
    centroid,
    model,
    top_n=2,
    max_articles=None
):
    """
    Retrieves articles closest to the period centroid.
    These are the most representative articles for that source-period.
    """

    source_df = df[df["source"] == source].copy()

    source_df["period_label"] = source_df["date"].apply(
        _period_from_date
    )

    period_df = source_df[
        source_df["period_label"] == period
    ].copy()

    period_df = period_df.dropna(
        subset=["text"]
    )

    if max_articles is not None and len(period_df) > max_articles:
        period_df = period_df.sample(
            n=max_articles,
            random_state=42
        )

    if period_df.empty:
        return []

    texts = period_df["text"].astype(str).tolist()

    article_embeddings = model.encode_documents(
        texts
    )

    scored = []

    for (_, row), article_embedding in zip(
        period_df.iterrows(),
        article_embeddings
    ):
        similarity = _cosine_similarity(
            article_embedding,
            centroid
        )

        scored.append(
            _article_record(
                row,
                similarity
            )
        )

    scored = sorted(
        scored,
        key=lambda x: x["similarity_to_period_centroid"],
        reverse=True
    )

    return scored[:top_n]


def build_representative_evidence_for_source(
    df,
    source,
    source_result,
    model,
    top_transitions=5,
    articles_per_period=2,
    max_articles_per_period=None
):
    """
    Builds representative article evidence for the highest-drift transitions.

    For each selected transition:
    - retrieves top representative articles before
    - retrieves top representative articles after
    """

    semantic = source_result.get("semantic_drift", {})

    labels = semantic.get("labels", [])
    values = semantic.get("values", [])
    threshold = semantic.get("threshold")

    if not labels or not values:
        return {
            "status": "skipped",
            "reason": "no semantic drift values",
            "transitions": []
        }

    transition_rows = []

    for label, value in zip(labels, values):
        transition_rows.append({
            "transition": label,
            "semantic_drift": float(value),
            "is_significant": (
                threshold is not None and value >= threshold
            )
        })

    transition_rows = sorted(
        transition_rows,
        key=lambda x: x["semantic_drift"],
        reverse=True
    )

    selected_transitions = transition_rows[:top_transitions]

    evidence_transitions = []

    for transition in selected_transitions:
        before_period, after_period = _split_transition(
            transition["transition"]
        )

        before_centroid = _get_period_centroid(
            source_result,
            before_period
        )

        after_centroid = _get_period_centroid(
            source_result,
            after_period
        )

        before_articles = []
        after_articles = []

        if before_centroid is not None:
            before_articles = retrieve_representative_articles_for_period(
                df=df,
                source=source,
                period=before_period,
                centroid=before_centroid,
                model=model,
                top_n=articles_per_period,
                max_articles=max_articles_per_period
            )

        if after_centroid is not None:
            after_articles = retrieve_representative_articles_for_period(
                df=df,
                source=source,
                period=after_period,
                centroid=after_centroid,
                model=model,
                top_n=articles_per_period,
                max_articles=max_articles_per_period
            )

        evidence_transitions.append({
            **transition,
            "before_period": before_period,
            "after_period": after_period,
            "before_articles": before_articles,
            "after_articles": after_articles
        })

    return {
        "status": "ok",
        "source": source,
        "selection_strategy": "highest semantic drift transitions; articles ranked by similarity to period centroid",
        "transitions": evidence_transitions
    }


def compute_evidence_score(evidence_result):
    """
    Scores how much article-level evidence is available and how representative it is.
    """

    if evidence_result.get("status") != "ok":
        return {
            "evidence_score": 0.0,
            "evidence_label": "low",
            "mean_article_similarity": 0.0,
            "num_evidence_articles": 0
        }

    similarities = []
    article_count = 0

    for transition in evidence_result.get("transitions", []):
        for article in transition.get("before_articles", []):
            similarities.append(
                article.get("similarity_to_period_centroid", 0.0)
            )
            article_count += 1

        for article in transition.get("after_articles", []):
            similarities.append(
                article.get("similarity_to_period_centroid", 0.0)
            )
            article_count += 1

    mean_similarity = (
        sum(similarities) / len(similarities)
        if similarities else 0.0
    )

    coverage_score = min(
        article_count / 20,
        1.0
    )

    evidence_score = (
        0.7 * mean_similarity
        + 0.3 * coverage_score
    )

    if evidence_score >= 0.70:
        label = "high"
    elif evidence_score >= 0.45:
        label = "medium"
    else:
        label = "low"

    return {
        "evidence_score": evidence_score,
        "evidence_label": label,
        "mean_article_similarity": mean_similarity,
        "num_evidence_articles": article_count
    }


def print_representative_evidence_summary(
    evidence_result,
    evidence_score=None,
    max_transitions=3
):
    print("\n=== REPRESENTATIVE EVIDENCE RETRIEVAL ===")

    if evidence_result.get("status") != "ok":
        print("Skipped:", evidence_result.get("reason"))
        return

    print("Source:", evidence_result.get("source"))
    print("Strategy:", evidence_result.get("selection_strategy"))

    if evidence_score:
        print(
            f"Evidence score: "
            f"{evidence_score['evidence_label']} "
            f"({evidence_score['evidence_score']:.3f})"
        )
        print(
            f"Mean article similarity: "
            f"{evidence_score['mean_article_similarity']:.3f}"
        )
        print(
            f"Evidence articles: "
            f"{evidence_score['num_evidence_articles']}"
        )

    for transition in evidence_result.get("transitions", [])[:max_transitions]:
        print(
            f"\nTransition: {transition['transition']} | "
            f"drift={transition['semantic_drift']:.4f} | "
            f"significant={transition['is_significant']}"
        )

        print("Before articles:")
        for article in transition.get("before_articles", []):
            print(
                f"- {article.get('title') or '[no title]'} "
                f"sim={article['similarity_to_period_centroid']:.3f}"
            )
            if article.get("url"):
                print(f"  URL: {article['url']}")

        print("After articles:")
        for article in transition.get("after_articles", []):
            print(
                f"- {article.get('title') or '[no title]'} "
                f"sim={article['similarity_to_period_centroid']:.3f}"
            )
            if article.get("url"):
                print(f"  URL: {article['url']}")