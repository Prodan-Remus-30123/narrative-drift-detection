ENTITY_BLACKLIST = {
    "reuters",
    "cnn",
    "bbc",
    "guardian",
    "new york times",
    "monday",
    "tuesday",
    "january",
    "february",
    "march",
    "april",
    "covid",
    "coronavirus",


    "getty images",
    "cnn health",
    "cnn healths",
    "iphone",
    "education healthcare",
    "states",
    "sign",
}

ALLOWED_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "GPE"
}

ENTITY_NORMALIZATION = {
    "u.s.": "united states",
    "us": "united states",
    "america": "united states",
    "uk": "united kingdom",
    "who": "world health organization"
}


MIN_ENTITY_GLOBAL_FREQUENCY = 3
MIN_VERB_COUNT = 5