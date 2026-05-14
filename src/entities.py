"""
entities.py

Entity-level framing analysis.
"""

import spacy
from collections import defaultdict, Counter


nlp = spacy.load("en_core_web_sm")


def analyze_entities(texts):
    """
    Analyze entity roles and associated verbs.

    Args:
        texts (list): List of documents

    Returns:
        dict: Entity statistics
    """

    entity_data = defaultdict(lambda: {
        "subject_count": 0,
        "object_count": 0,
        "verbs": Counter()
    })

    for text in texts:

        doc = nlp(text)

        for token in doc:

            # Ignore tokens that are not part of named entities
            if token.ent_type_ == "":
                continue

            entity = token.text

            # Subject roles
            if token.dep_ in ("nsubj", "nsubjpass"):
                entity_data[entity]["subject_count"] += 1

            # Object roles
            if token.dep_ in ("dobj", "pobj", "obj"):
                entity_data[entity]["object_count"] += 1

            # Associated verbs
            if token.head.pos_ == "VERB":
                verb = token.head.lemma_.lower()
                entity_data[entity]["verbs"][verb] += 1

    return entity_data