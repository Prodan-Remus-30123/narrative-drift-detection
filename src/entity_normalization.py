"""
entity_normalization.py

Entity normalization utilities.
"""

import re


COMMON_PREFIXES = {

    "the",
    "dr",
    "prof",
    "president",
    "mr",
    "mrs",
    "ms"
}


MANUAL_ALIASES = {
    "who": "world health organization",
    "covid": "covid-19",
    "covid19": "covid-19",
    "donald trump": "trump",
    "president trump": "trump",
    "u.s.": "united states",
    "us": "united states",
}


def normalize_entity(entity):

    entity = entity.lower()

    entity = re.sub(
        r"[^\w\s-]",
        "",
        entity
    )

    words = entity.split()

    words = [

        word for word in words

        if word not in COMMON_PREFIXES
    ]

    entity = " ".join(words)
    
    if entity.endswith(" who"):
        entity = entity.replace(
            " who",
            ""
        )
    
    if entity.endswith("s"):

        singular_candidate = entity[:-1]

        if singular_candidate in {
            "hong kong"
        }:

            entity = singular_candidate

    entity = entity.strip()

    if entity in MANUAL_ALIASES:

        entity = MANUAL_ALIASES[
            entity
        ]

    return entity