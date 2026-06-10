from agentic_llm.llm_client import OllamaClient
from agentic_llm.planner_agent import PlannerAgent
from agentic_llm.semantic_agent import SemanticAgent
from agentic_llm.framing_agent import FramingAgent
from agentic_llm.affective_agent import AffectiveAgent
from agentic_llm.synthesis_agent import SynthesisAgent
from agentic_llm.validator_agent import ValidatorAgent


class AgenticNarrativeRunner:
    def __init__(self, model="qwen2.5:7b"):
        self.llm = OllamaClient(model=model)

        self.planner = PlannerAgent(self.llm)
        self.semantic = SemanticAgent(self.llm)
        self.framing = FramingAgent(self.llm)
        self.affective = AffectiveAgent(self.llm)
        self.synthesis = SynthesisAgent(self.llm)
        self.validator = ValidatorAgent()

    def answer(self, question, packet):
        warnings = self.validator.validate_packet(packet)

        selected_agents = self.planner.plan(
            question=question,
            packet=packet
        )

        print("\n=== PLANNER DECISION ===")
        print("Question:", question)
        print("Selected agents:", selected_agents)

        specialist_outputs = {}

        if "semantic" in selected_agents:
            specialist_outputs["semantic"] = self.semantic.explain(
                question,
                packet
            )

        if "framing" in selected_agents:
            specialist_outputs["framing"] = self.framing.explain(
                question,
                packet
            )

        if "affective" in selected_agents:
            specialist_outputs["affective"] = self.affective.explain(
                question,
                packet
            )

        final_answer = self.synthesis.synthesize(
            question=question,
            packet=packet,
            specialist_outputs=specialist_outputs
        )

        return {
            "selected_agents": selected_agents,
            "warnings": warnings,
            "specialist_outputs": specialist_outputs,
            "final_answer": final_answer
        }