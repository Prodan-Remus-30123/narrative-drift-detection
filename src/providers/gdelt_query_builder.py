"""
gdelt_query_builder.py

Utilities for building raw GDELT queries.
"""


def build_boolean_query(
    required_terms,
    optional_terms=None,
    excluded_terms=None
):

    query_parts = []

    # REQUIRED TERMS

    if required_terms:

        required = " AND ".join(
            required_terms
        )

        query_parts.append(
            required
        )

    # OPTIONAL TERMS

    if optional_terms:

        optional = " OR ".join(
            optional_terms
        )

        query_parts.append(
            f"({optional})"
        )

    # EXCLUDED TERMS

    if excluded_terms:

        excluded = " OR ".join(
            excluded_terms
        )

        query_parts.append(
            f"NOT ({excluded})"
        )

    return " AND ".join(
        query_parts
    )