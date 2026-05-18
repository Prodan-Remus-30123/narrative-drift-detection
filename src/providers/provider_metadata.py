"""
provider_metadata.py

Provider capabilities and retrieval metadata.
"""

PROVIDER_METADATA = {

    "GDELTProvider": {

        "supports_historical": True,

        "supports_domains": True,

        "preferred_query_style": "broad",

        "priority": 1
    },

    "GuardianProvider": {

        "supports_historical": True,

        "supports_domains": False,

        "preferred_query_style": "precise",

        "priority": 2
    }
}