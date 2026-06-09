import re


class ValidatorAgent:

    def validate_packet(self, packet, specialist_outputs=None):

        warnings = []

        entities = packet.get(
            "who",
            {}
        ).get(
            "top_entities",
            []
        )

        allowed_entities = {
            e["entity"].lower()
            for e in entities
        }

        # ------------------
        # DATA QUALITY
        # ------------------

        for entity in entities:

            name = entity.get(
                "entity",
                "unknown"
            )

            turnover = entity.get(
                "vocabulary_turnover"
            )

            similarity = entity.get(
                "shared_similarity"
            )

            added = entity.get(
                "added_verbs",
                []
            )

            removed = entity.get(
                "removed_verbs",
                []
            )

            verbs = (
                entity.get("before_verbs", [])
                + entity.get("after_verbs", [])
                + added
                + removed
            )

            bad_verbs = [
                v
                for v in verbs
                if len(str(v).strip()) <= 1
            ]

            if bad_verbs:
                warnings.append(
                    f"{name}: suspicious verbs {bad_verbs}"
                )

            changed_count = (
                len(set(added))
                + len(set(removed))
            )

            if (
                turnover is not None
                and turnover >= 0.8
                and changed_count < 2
            ):
                warnings.append(
                    f"{name}: very high turnover "
                    f"but very few verb changes"
                )

            if (
                similarity is None
                and changed_count >= 5
            ):
                warnings.append(
                    f"{name}: no shared verbs "
                    f"between periods"
                )

        # ------------------
        # OUTPUT VALIDATION
        # ------------------

        if specialist_outputs:

            combined_text = "\n".join(
                specialist_outputs.values()
            )

            words = re.findall(
                r"[A-Z][A-Za-z\\-]+(?:\\s+[A-Z][A-Za-z\\-]+)*",
                combined_text
            )

            hallucinated = []

            for candidate in words:

                candidate_lower = candidate.lower()

                if (
                    candidate_lower not in allowed_entities
                    and candidate_lower not in {
                        "semantic",
                        "framing",
                        "affective"
                    }
                ):
                    hallucinated.append(candidate)

            hallucinated = sorted(
                set(hallucinated)
            )

            if hallucinated:
                warnings.append(
                    "Possible hallucinated entities: "
                    + ", ".join(hallucinated[:10])
                )

        return warnings