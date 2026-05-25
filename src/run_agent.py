from agents.semantic_drift_agent import (
    SemanticDriftAgent
)

from agents.entity_framing_agent import (
    EntityFramingAgent
)

from agents.affective_dynamics_agent import (
    AffectiveDynamicsAgent
)

from agents.conflict_trust_agent import (
    ConflictTrustAgent
)

from agents.editorial_behavior_agent import (
    EditorialBehaviorAgent
)

from agents.evidence_validator_agent import (
    EvidenceValidatorAgent
)

from agents.narrative_summary_agent import (
    NarrativeSummaryAgent
)

from agents.orchestrator_agent import NarrativeOrchestratorAgent
from agentic_tools.context_registry import (
    clear_context
)

clear_context()

orchestrator = NarrativeOrchestratorAgent()

result = orchestrator.answer(
    query="How did CNN portray China during COVID?",
    source="cnn.com",
    entity="china"
)

print(result["plan"])
print(result["final_answer"]["summary"])
print("Overall confidence:", result["final_answer"]["overall_confidence"])

# source = "cnn.com"
# entity = "china"

# semantic_agent = SemanticDriftAgent()
# framing_agent = EntityFramingAgent()
# affective_agent = AffectiveDynamicsAgent()
# conflict_agent = ConflictTrustAgent()
# editorial_agent = EditorialBehaviorAgent()

# validator = EvidenceValidatorAgent()

# summary_agent = NarrativeSummaryAgent()


# semantic_result = semantic_agent.analyze(
#     source
# )

# framing_result = framing_agent.analyze(
#     source,
#     entity
# )

# affective_result = affective_agent.analyze(
#     source
# )

# conflict_result = conflict_agent.analyze(
#     source
# )

# editorial_result = editorial_agent.analyze(
#     source
# )

# validation_result = validator.validate(
#     framing_result
# )

# summary = summary_agent.summarize(

#     source=source,

#     semantic_result=semantic_result,

#     framing_result=framing_result,

#     affective_result=affective_result,

#     conflict_result=conflict_result,

#     editorial_result=editorial_result,

#     validation_result=validation_result
# )

# print(summary["summary"])
# print("\nOverall confidence:", summary["overall_confidence"])