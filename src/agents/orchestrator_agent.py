"""
orchestrator_agent.py

Narrative Orchestrator Agent.

Routes user questions to specialized agents,
collects outputs, validates evidence, and produces
a final narrative summary.
"""

from agents.semantic_drift_agent import SemanticDriftAgent
from agents.entity_framing_agent import EntityFramingAgent
from agents.affective_dynamics_agent import AffectiveDynamicsAgent
from agents.conflict_trust_agent import ConflictTrustAgent
from agents.editorial_behavior_agent import EditorialBehaviorAgent
from agents.evidence_validator_agent import EvidenceValidatorAgent
from agents.narrative_summary_agent import NarrativeSummaryAgent


class NarrativeOrchestratorAgent:

    def __init__(self):
        self.name = "Narrative Orchestrator Agent"

        self.semantic_agent = SemanticDriftAgent()
        self.framing_agent = EntityFramingAgent()
        self.affective_agent = AffectiveDynamicsAgent()
        self.conflict_agent = ConflictTrustAgent()
        self.editorial_agent = EditorialBehaviorAgent()

        self.validator_agent = EvidenceValidatorAgent()
        self.summary_agent = NarrativeSummaryAgent()

    def answer(
        self,
        query,
        source,
        entity=None
    ):
        plan = self._plan(
            query=query,
            entity=entity
        )

        outputs = {}

        if plan["semantic"]:
            outputs["semantic"] = self.semantic_agent.analyze(
                source=source
            )

        if plan["framing"] and entity is not None:
            outputs["framing"] = self.framing_agent.analyze(
                source=source,
                entity=entity
            )

        if plan["affective"]:
            outputs["affective"] = self.affective_agent.analyze(
                source=source
            )

        if plan["conflict"]:
            outputs["conflict"] = self.conflict_agent.analyze(
                source=source
            )

        if plan["editorial"]:
            outputs["editorial"] = self.editorial_agent.analyze(
                source=source
            )

        validation = None

        if "framing" in outputs:
            validation = self.validator_agent.validate(
                outputs["framing"]
            )

        summary = self.summary_agent.summarize(
            source=source,
            semantic_result=outputs.get("semantic"),
            framing_result=outputs.get("framing"),
            affective_result=outputs.get("affective"),
            conflict_result=outputs.get("conflict"),
            editorial_result=outputs.get("editorial"),
            validation_result=validation
        )

        return {
            "agent": self.name,
            "query": query,
            "source": source,
            "entity": entity,
            "plan": plan,
            "outputs": outputs,
            "validation": validation,
            "final_answer": summary
        }

    def _plan(
        self,
        query,
        entity=None
    ):
        query_lower = query.lower()

        plan = {
            "semantic": False,
            "framing": False,
            "affective": False,
            "conflict": False,
            "editorial": False
        }

        # Always useful for broad narrative questions
        if any(word in query_lower for word in [
            "narrative",
            "changed",
            "evolved",
            "shift",
            "drift",
            "portray",
            "framed",
            "coverage"
        ]):
            plan["semantic"] = True
            plan["framing"] = entity is not None
            plan["editorial"] = True

        if any(word in query_lower for word in [
            "sentiment",
            "emotional",
            "tone",
            "negative",
            "positive",
            "fear",
            "anger"
        ]):
            plan["affective"] = True

        if any(word in query_lower for word in [
            "conflict",
            "trust",
            "polarization",
            "accuse",
            "blame",
            "institutional"
        ]):
            plan["conflict"] = True

        # If user asks about an entity, framing is usually needed.
        if entity is not None:
            plan["framing"] = True

        # For now, editorial behavior helps contextualize most analyses.
        if any(word in query_lower for word in [
            "cnn",
            "bbc",
            "guardian",
            "outlet",
            "source",
            "editorial"
        ]):
            plan["editorial"] = True

        # Fallback: if query is vague, run the core stack.
        if not any(plan.values()):
            plan["semantic"] = True
            plan["editorial"] = True

            if entity is not None:
                plan["framing"] = True

        return plan