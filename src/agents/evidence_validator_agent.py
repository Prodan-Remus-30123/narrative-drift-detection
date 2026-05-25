"""
evidence_validator_agent.py

Checks whether agent interpretations are supported
by numeric metrics and retrieved evidence.
"""


class EvidenceValidatorAgent:

    def __init__(self):
        self.name = "Evidence Validator Agent"

    def validate(self, agent_output):
        status = agent_output.get("status", "unknown")

        if status != "ok":
            return {
                "agent": self.name,
                "validated_agent": agent_output.get("agent"),
                "status": "not_validated",
                "verdict": "Cannot validate because the source agent did not return ok status.",
                "confidence": "low",
                "flags": ["source_agent_not_ok"]
            }

        flags = []

        numeric_support = self._has_numeric_support(agent_output)
        evidence_support = self._has_evidence_support(agent_output)

        if not numeric_support:
            flags.append("missing_numeric_support")

        if not evidence_support:
            flags.append("missing_textual_evidence")

        if numeric_support and evidence_support:
            verdict = "supported"
            confidence = "high"
        elif numeric_support:
            verdict = "partially_supported"
            confidence = "medium"
        else:
            verdict = "weakly_supported"
            confidence = "low"

        return {
            "agent": self.name,
            "validated_agent": agent_output.get("agent"),
            "status": "ok",
            "verdict": verdict,
            "confidence": confidence,
            "flags": flags,
            "numeric_support": numeric_support,
            "evidence_support": evidence_support,
            "original_interpretation": agent_output.get("interpretation", "")
        }

    def _has_numeric_support(self, agent_output):
        numeric_fields = [
            "semantic_results",
            "framing_results",
            "sentiment_results",
            "conflict_trust_results",
            "profile",
            "emotional_volatility",
            "average_compound"
        ]

        for field in numeric_fields:
            if field in agent_output and agent_output[field]:
                return True

        return False

    def _has_evidence_support(self, agent_output):
        evidence = agent_output.get("evidence")

        if not evidence:
            return False

        evidence_by_period = evidence.get("evidence", {})

        total_snippets = 0

        for _, snippets in evidence_by_period.items():
            total_snippets += len(snippets)

        return total_snippets > 0