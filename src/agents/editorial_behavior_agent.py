"""
editorial_behavior_agent.py

Agentic Editorial Behavior Agent.

This agent profiles a news source by combining:
- semantic drift
- framing drift
- affective dynamics
- conflict/trust signals
"""

from agentic_tools.editorial_behavior_tools import (
    get_editorial_behavior_profile
)
from entity_latent_frames import (
    compute_entity_latent_frame_transitions
)

from actor_graph import (
    compute_actor_graph_centrality
)


class EditorialBehaviorAgent:

    def __init__(self):
        self.name = "Editorial Behavior Agent"

    def analyze(self, source):
        profile = get_editorial_behavior_profile(source)
        actor_graph = compute_actor_graph_centrality(source)

        top_actors = [actor for actor, score in actor_graph["actor_centrality"][:3]]

        latent_ecosystems = {}

        for actor in top_actors:
            try:
                latent_ecosystems[actor] = (compute_entity_latent_frame_transitions(source=source, entity=actor))

            except Exception as error:

                print(f"Latent ecosystem error " f"for {actor}: {error}")
        interpretation = self._interpret(profile, latent_ecosystems)
        label = self._classify_behavior(profile)
        confidence = self._estimate_confidence(profile)

        return {
            "agent": self.name,
            "source": source,
            "status": "ok",
            "editorial_label": label,
            "interpretation": interpretation,
            "profile": profile,
            "confidence": confidence,
            "latent_ecosystems": latent_ecosystems
        }

    def _classify_behavior(self, profile):
        semantic = profile["avg_semantic_drift"]
        framing = profile["avg_framing_drift"]
        emotional_volatility = profile["emotional_volatility"]
        conflict = profile["avg_conflict_score"]
        trust = profile["avg_trust_score"]

        if semantic < 0.06 and framing > 0.65:
            if conflict > 0.035:
                return "stable-topic conflictual reframing"
            return "stable-topic high reframing"

        if semantic >= 0.06 and framing > 0.65:
            return "topic-shifting high reframing"

        if framing < 0.45 and emotional_volatility < 0.12:
            return "stable institutional reporting"

        if trust < -0.2:
            return "institutional trust erosion"

        return "mixed editorial behavior"

    def _interpret(self, profile, latent_ecosystems):
        source = profile["source"]

        interpretation = (
            f"{source} shows an average semantic drift of "
            f"{profile['avg_semantic_drift']:.3f} and an average "
            f"framing drift of {profile['avg_framing_drift']:.3f}. "
        )

        if profile["avg_semantic_drift"] < 0.06:
            interpretation += (
                "This suggests broad topic stability across the observed periods. "
            )
        else:
            interpretation += (
                "This suggests stronger topic-level movement across periods. "
            )

        if profile["avg_framing_drift"] > 0.65:
            interpretation += (
                "At the same time, actor-level framing changes are substantial, "
                "indicating that the source keeps covering a similar topic while "
                "changing how key actors are represented. "
            )
        else:
            interpretation += (
                "Actor-level framing changes are relatively moderate. "
            )

        interpretation += (
            f"The source has {profile['major_reframing_events']} major "
            f"reframing events and {profile['significant_semantic_shifts']} "
            f"significant semantic shifts. "
        )

        interpretation += (
            f"Its average sentiment is {profile['avg_sentiment']:.3f}, "
            f"with emotional volatility {profile['emotional_volatility']:.3f}. "
        )

        interpretation += (
            f"Average conflict score is {profile['avg_conflict_score']:.3f}, "
            f"average polarization score is {profile['avg_polarization_score']:.3f}, "
            f"and average institutional trust score is "
            f"{profile['avg_trust_score']:.3f}."
        )

        ecosystem_summaries = []

        for actor, data in latent_ecosystems.items():
            transitions = data["latent_frame_transitions"]

            if len(transitions) == 0:
                continue

            strongest = transitions[0]

            ecosystem_summaries.append(
                f"{actor} evolves from "
                f"{strongest['before_frame']} "
                f"toward "
                f"{strongest['after_frame']}"
            )

        if len(ecosystem_summaries) > 0:
            interpretation += (" Dominant narrative ecosystems include: " + "; ".join(ecosystem_summaries[:3]) + ".")

        return interpretation

    def _estimate_confidence(self, profile):
        semantic_periods = len(
            profile["semantic_data"]["semantic_drift"]
        )

        conflict_periods = len(
            profile["conflict_data"]["conflict_trust_results"]
        )

        if semantic_periods >= 4 and conflict_periods >= 4:
            return "high"

        if semantic_periods >= 2:
            return "medium"

        return "low"