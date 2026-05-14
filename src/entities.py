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

        for ent in doc.ents:

            # Ignore tokens that are not part of named entities
            allowed_entity_types = {
                "ORG",
                "GPE",
                "PERSON"
            }

            if ent.label_ not in allowed_entity_types:
                continue

            entity = ent.text
            if len(entity) < 3:
                continue

            ignored_entities = {
                "month",
                "year",
                "day",
                "time"
            }

            if entity.lower() in ignored_entities:
                continue

            # Subject roles
            root = ent.root
            dep = root.dep_
            if dep in ("nsubj", "nsubjpass"):
                entity_data[entity]["subject_count"] += 1

            # Object roles
            if dep in ("dobj", "pobj", "obj"):
                entity_data[entity]["object_count"] += 1

            # Associated verbs
            if root.head.pos_ == "VERB":
                verb = root.head.lemma_.lower()
                entity_data[entity]["verbs"][verb] += 1

    return entity_data