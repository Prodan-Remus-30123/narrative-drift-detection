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
    "coronavirus"
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

SEMANTICALLY_WEAK_VERBS = {
    "come",
    "go",
    "work",
    "include",
    "show",
    "give",
    "take",
    "make",
    "begin",
    "continue",
    "need",
    "think",
    "see",
    "appear",
    "become",
    "put",
    "add",
    "call",
    "say",
    "use",
    "ask"
}

GENERIC_VERBS = {
                "be",
                "have",
                "do",
                "say",
                "make",
                "go",
                "take",
                "use",
                "tell",
                "report"
            }

MIN_ENTITY_GLOBAL_FREQUENCY = 3
MIN_VERB_COUNT = 5