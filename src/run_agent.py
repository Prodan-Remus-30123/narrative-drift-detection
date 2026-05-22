from agentic_tools.semantic_tools import get_semantic_drift
from agentic_tools.framing_tools import (
    get_entity_framing
)
from agentic_tools.sentiment_tools import (
    get_sentiment_evolution
)
from agentic_tools.salience_tools import (
    get_actor_salience
)

from agentic_tools.evidence_tools import (
    retrieve_entity_evidence
)

from agents.entity_framing_agent import (
    EntityFramingAgent
)


agent = EntityFramingAgent()

result = agent.analyze(
    source="cnn.com",
    entity="china"
)

print(result["interpretation"])
print("Confidence:", result["confidence"])

# result = retrieve_entity_evidence(
#     source="cnn.com",
#     entity="china",
#     verbs=[
#         "accuse",
#         "criticize",
#         "restrict"
#     ],
#     max_snippets_per_period=2
# )

# print(result)

# result = get_actor_salience(
#     source="cnn.com",
#     entity="china"
# )

# print(result)

# result = get_sentiment_evolution(
#     "cnn.com"
# )

# print(result)

# result = get_entity_framing(
#     source="cnn.com",
#     entity="china"
# )

# print(result)

# print(get_semantic_drift("cnn.com"))