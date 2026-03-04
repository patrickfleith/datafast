"""Branch with Rewrite steps + JoinBranches with custom suffixes.

Demonstrates: Branching to create multiple style variations of the same text,
then merging with custom suffixes.
"""

from datafast_v2 import Source, Rewrite, Branch, JoinBranches, Sink, openrouter

import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

pipeline = (
    Source.list([
        {"text": "Machine learning models can identify patterns in large datasets."},
        {"text": "The process of photosynthesis converts light energy into chemical energy."},
    ])
    >> Branch(
        formal=Rewrite(input_column="text", llm=model, mode="formalize"),
        casual=Rewrite(input_column="text", llm=model, mode="informalize"),
    )
    >> JoinBranches(suffixes={"formal": "_formal", "casual": "_casual"})
    >> Sink.jsonl("v2_examples/outputs/35_branch_rewrite_styles.jsonl")
)

records = pipeline.run()
