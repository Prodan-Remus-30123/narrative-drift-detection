"""
entity_framing_agent.py

Agentic Entity Framing Agent.

This agent investigates how a specific entity
is framed over time by a news source.
"""

from agentic_tools.framing_tools import (
    get_entity_framing
)

from agentic_tools.salience_tools import (
    get_actor_salience
)

from agentic_tools.evidence_tools import (
    retrieve_entity_evidence
)
from entity_latent_frames import compute_entity_latent_frame_transitions



class EntityFramingAgent:

    def __init__(self):
        self.name = "Entity Framing Agent"

    def analyze(
        self,
        source,
        entity,
        max_evidence_per_period=2
    ):
        """
        Analyze how an entity is reframed over time.

        Args:
            source (str)
            entity (str)

        Returns:
            dict
        """

        framing_data = get_entity_framing(
            source=source,
            entity=entity
        )

        salience_data = get_actor_salience(
            source=source,
            entity=entity
        )

        framing_results = framing_data["framing_results"]
        latent_frames = compute_entity_latent_frame_transitions(source=source,entity=entity)

        if len(framing_results) == 0:

            return {
                "agent": self.name,
                "source": source,
                "entity": entity,
                "status": "insufficient_data",
                "interpretation": (
                    "No reliable framing drift data "
                    "was found for this entity."
                ),
                "evidence": {},
                "confidence": "low"
            }

        evidence_verbs = self._select_evidence_verbs(
            framing_results
        )

        evidence_data = retrieve_entity_evidence(
            source=source,
            entity=entity,
            verbs=evidence_verbs,
            max_snippets_per_period=max_evidence_per_period
        )

        interpretation = self._interpret(
            source=source,
            entity=entity,
            framing_results=framing_results,
            salience_data=salience_data,
            latent_frames=latent_frames
        )

        confidence = self._estimate_confidence(
            framing_results,
            evidence_data
        )

        return {
            "agent": self.name,
            "source": source,
            "entity": entity,
            "status": "ok",
            "interpretation": interpretation,
            "framing_results": framing_results,
            "salience": salience_data,
            "evidence_verbs": evidence_verbs,
            "evidence": evidence_data,
            "latent_frames": latent_frames,
            "confidence": confidence
        }

    def _select_evidence_verbs(
        self,
        framing_results,
        top_n=8
    ):
        """
        Select important verbs for evidence retrieval.
        """

        verbs = []

        for result in framing_results:

            verbs.extend(
                result["before_verbs"][:3]
            )

            verbs.extend(
                result["after_verbs"][:3]
            )

        unique_verbs = []

        for verb in verbs:

            if verb not in unique_verbs:
                unique_verbs.append(verb)

        return unique_verbs[:top_n]

    def _interpret(
        self,
        source,
        entity,
        framing_results,
        salience_data,
        latent_frames
    ):
        """
        Produce a grounded first-pass interpretation.
        This is rule-based for now.
        Later this can be replaced by an LLM call.
        """

        major_shifts = [
            item for item in framing_results
            if item["drift_class"] == "major reframing"
        ]

        avg_drift = (
            sum(item["drift"] for item in framing_results)
            /
            len(framing_results)
        )

        first = framing_results[-1]
        strongest = framing_results[0]
        latent_transitions = latent_frames["latent_frame_transitions"]

        strongest_latent = latent_transitions[0]
        salience_items = salience_data.get(
            "actor_salience",
            []
        )

        if salience_items:
            total_salience = salience_items[0][
                "total_salience"
            ]
            periods_present = salience_items[0][
                "periods_present"
            ]
        else:
            total_salience = 0
            periods_present = 0

        interpretation = (
            f"{entity} shows an average framing drift "
            f"of {avg_drift:.3f} in {source}. "
        )

        if len(major_shifts) > 0:
            interpretation += (
                f"The entity undergoes repeated major "
                f"reframing events across "
                f"{len(major_shifts)} transitions. "
            )
        else:
            interpretation += (
                "The entity shows mostly stable or moderate "
                "framing changes. "
            )

        interpretation += (
            f"Its total salience is {total_salience:.1f}, "
            f"and it appears across {periods_present} periods. "
        )

        interpretation += (
            f"The strongest observed reframing occurs during "
            f"{strongest['transition']}, where the narrative "
            f"shifts from "
            f"{strongest_latent['before_frame']} "
            f"toward "
            f"{strongest_latent['after_frame']}."
        )

        return interpretation

    def _estimate_confidence(
        self,
        framing_results,
        evidence_data
    ):
        """
        Estimate confidence from number of framing transitions
        and available evidence snippets.
        """

        num_transitions = len(
            framing_results
        )

        evidence_count = 0

        for period, snippets in evidence_data.get(
            "evidence",
            {}
        ).items():

            evidence_count += len(
                snippets
            )

        if num_transitions >= 3 and evidence_count >= 4:
            return "high"

        if num_transitions >= 2 and evidence_count >= 2:
            return "medium"

        return "low"