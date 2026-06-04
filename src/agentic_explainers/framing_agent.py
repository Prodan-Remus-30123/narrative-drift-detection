class FramingAgent:
    def explain(self, packet):
        entities = packet.get("who", {}).get("top_entities", [])

        lines = []
        lines.append(
            f"FramingAgent analysis for {packet['source']} | {packet['transition']}"
        )
        lines.append("")
        lines.append("Actor-level reframing:")

        if not entities:
            lines.append("- No entity reframing evidence available.")
            return "\n".join(lines)

        for entity in entities:
            name = entity.get("entity", "unknown entity")
            drift = entity.get("drift", 0.0)
            role = entity.get("ecosystem_role", "unknown role")

            before = entity.get("before_verbs", [])
            after = entity.get("after_verbs", [])
            added = entity.get("added_verbs", [])
            removed = entity.get("removed_verbs", [])

            lines.append("")
            lines.append(f"- {name}")
            lines.append(f"  - Framing drift: {drift:.4f}")
            lines.append(f"  - Ecosystem role: {role}")
            lines.append(f"  - Earlier verbs: {', '.join(before[:8])}")
            lines.append(f"  - Later verbs: {', '.join(after[:8])}")
            lines.append(f"  - Added verbs: {', '.join(added[:8])}")
            lines.append(f"  - Removed verbs: {', '.join(removed[:8])}")

        lines.append("")
        lines.append(
            "Evidence boundary: this agent explains actor-level reframing from verb evidence only."
        )

        return "\n".join(lines)


def explain_framing(packet):
    agent = FramingAgent()
    return agent.explain(packet)