"""
conflict_trust_agent.py

Agentic Conflict & Trust Agent.

Analyzes conflict escalation, institutional trust,
and political polarization signals.
"""

from agentic_tools.conflict_trust_tools import (
    get_conflict_trust_signals
)
from entity_latent_frames import (
    compute_entity_latent_frame_transitions
)

from actor_graph import (
    compute_actor_graph_centrality
)


class ConflictTrustAgent:

    def __init__(self):
        self.name = "Conflict & Trust Agent"

    def analyze(self, source):

        signal_data = get_conflict_trust_signals(source=source)

        results = signal_data["conflict_trust_results"]
        actor_graph = compute_actor_graph_centrality(source)

        top_actors = [actor for actor, score in actor_graph["actor_centrality"][:5]]

        if len(results) == 0:
            return {
                "agent": self.name,
                "source": source,
                "status": "insufficient_data",
                "interpretation": (
                    "No conflict or trust data was available."
                ),
                "confidence": "low"
            }
        
        latent_conflict_frames = {}

        for actor in top_actors:
            try:

                latent_conflict_frames[actor] = (compute_entity_latent_frame_transitions(source=source,entity=actor))

            except Exception as error:

                print(
                    f"Conflict frame error for "
                    f"{actor}: {error}"
                )

        interpretation = self._interpret(source=source, results=results, latent_conflict_frames=latent_conflict_frames)

        confidence = self._estimate_confidence(results)

        return {
            "agent": self.name,
            "source": source,
            "status": "ok",
            "interpretation": interpretation,
            "conflict_trust_results": results,
            "confidence": confidence,
            "latent_conflict_frames": latent_conflict_frames
        }

    def _interpret(self, source, results, latent_conflict_frames):

        periods = list(results.keys())

        first = periods[0]
        last = periods[-1]

        first_conflict = results[first]["conflict_score"]
        last_conflict = results[last]["conflict_score"]

        conflict_change = last_conflict - first_conflict

        max_conflict_period = max(
            results.items(),
            key=lambda x: x[1]["conflict_score"]
        )

        max_polarization_period = max(
            results.items(),
            key=lambda x: x[1]["polarization_score"]
        )

        lowest_trust_period = min(
            results.items(),
            key=lambda x: x[1]["trust_score"]
        )

        interpretation = (
            f"{source} shows a conflict score change "
            f"from {first_conflict:.3f} in {first} "
            f"to {last_conflict:.3f} in {last}. "
        )

        if conflict_change > 0.02:
            interpretation += (
                "This suggests increasing conflict framing "
                "over time. "
            )
        elif conflict_change < -0.02:
            interpretation += (
                "This suggests decreasing conflict framing "
                "over time. "
            )
        else:
            interpretation += (
                "Conflict framing remains relatively stable "
                "over time. "
            )

        interpretation += (
            f"The highest conflict period is "
            f"{max_conflict_period[0]} "
            f"with score {max_conflict_period[1]['conflict_score']:.3f}. "
        )

        interpretation += (
            f"The strongest political polarization signal occurs in "
            f"{max_polarization_period[0]} "
            f"with score {max_polarization_period[1]['polarization_score']:.3f}. "
        )

        interpretation += (
            f"The lowest institutional trust signal occurs in "
            f"{lowest_trust_period[0]} "
            f"with trust score {lowest_trust_period[1]['trust_score']:.3f}."
        )

        frame_summaries = []
        ecosystem_types = []

        for actor, data in latent_conflict_frames.items():
            transitions = data["latent_frame_transitions"]

            if len(transitions) == 0:
                continue

            strongest = transitions[0]

            after_frame = strongest["after_frame"]
            if after_frame not in ecosystem_types:
                ecosystem_types.append(after_frame)

            frame_summaries.append(f"{actor}: {after_frame}")

        if len(ecosystem_types) > 0:
            interpretation += (
                " Dominant conflict ecosystems include: "
                + "; ".join(
                    list(ecosystem_types)[:5]
                )
                + "."
            )

        return interpretation

    def _estimate_confidence(self, results):

        num_periods = len(results)

        total_verbs = sum(
            period_data["total_verbs"]
            for period_data in results.values()
        )

        if num_periods >= 5 and total_verbs >= 500:
            return "high"

        if num_periods >= 3 and total_verbs >= 200:
            return "medium"

        return "low"