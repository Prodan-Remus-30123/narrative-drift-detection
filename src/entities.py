"""
entities.py

Entity-level framing analysis.
"""

import spacy
from collections import defaultdict, Counter
from entity_normalization import (normalize_entity)
from filters.entity_filters import (ENTITY_BLACKLIST)
from filters.verb_filters import (SEMANTICALLY_WEAK_VERBS, GENERIC_VERBS)

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

            entity = normalize_entity(ent.text)

            # if entity.islower() and len(entity.split()) == 1:
            #     continue



            if len(entity.split()) == 1 and len(entity) < 4:
                continue

            if entity in ENTITY_BLACKLIST:
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

            verb = None

            current = root

            while current.head != current:

                current = current.head

                if current.pos_ == "VERB":

                    verb = current.lemma_.lower()

                    if not verb.isalpha():
                        verb = None
                        break

                    break

            if verb in GENERIC_VERBS:
                continue

            # if verb in SEMANTICALLY_WEAK_VERBS:
            #     continue

            if verb:
                entity_data[entity]["verbs"][verb] += 1

    MIN_LOCAL_VERB_FREQUENCY = 1

    filtered_entity_data = {}

    for entity, stats in entity_data.items():

        filtered_verbs = Counter()

        for verb, count in stats["verbs"].items():

            if count >= MIN_LOCAL_VERB_FREQUENCY:
                filtered_verbs[verb] = count

        if len(filtered_verbs) == 0:
            continue

        stats["verbs"] = filtered_verbs

        filtered_entity_data[entity] = stats

    return filtered_entity_data