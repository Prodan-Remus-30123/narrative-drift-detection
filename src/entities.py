"""
entities.py

Entity-level framing analysis.
"""

import spacy
from collections import defaultdict, Counter
from entity_normalization import (normalize_entity)
from filters.entity_filters import (ENTITY_BLACKLIST)
from filters.verb_filters import (SEMANTICALLY_WEAK_VERBS, GENERIC_VERBS)
import json
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

MIN_VERB_LENGTH = 3

LEMMA_CORRECTIONS_PATH = Path(__file__).parent / "lemma_corrections.json"


def load_lemma_corrections():
    if not LEMMA_CORRECTIONS_PATH.exists():
        return {}

    with open(LEMMA_CORRECTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        key: value
        for key, value in data.items()
        if value is not None
    }


LEMMA_CORRECTIONS = load_lemma_corrections()

def get_valid_framing_lemma(token):
    lemma = token.lemma_.lower().strip()

    lemma = LEMMA_CORRECTIONS.get(
        lemma,
        lemma
    )

    if token.pos_ != "VERB":
        return None

    if token.tag_ in {"MD"}:
        return None

    if not lemma.isalpha():
        return None

    if len(lemma) < MIN_VERB_LENGTH:
        return None

    if lemma in GENERIC_VERBS:
        return None

    if lemma in SEMANTICALLY_WEAK_VERBS:
        return None

    return lemma

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

                    verb = get_valid_framing_lemma(current)

                    if verb is None:
                        break

                    break

            # GENERIC_VERBS/SEMANTICALLY_WEAK_VERBS are already excluded
            # by get_valid_framing_lemma above, so `verb` here is never
            # a member of either set.
            if verb:
                entity_data[entity]["verbs"][verb] += 1

    return {
        entity: stats
        for entity, stats in entity_data.items()
        if stats["verbs"]
    }