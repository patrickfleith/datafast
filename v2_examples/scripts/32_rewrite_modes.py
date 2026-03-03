"""Rewrite step: multiple modes — simplify, formalize, custom, audience.

Demonstrates: Different Rewrite modes with mode-specific parameters.
"""

from datafast_v2 import Source, Rewrite, Sink, openrouter

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

data = [
    {"text": "The mitochondrial electron transport chain facilitates oxidative phosphorylation via a proton gradient."},
]
source = Source.list(data)

# Simplify for a specific audience
pipeline_simplify = (
    source
    >> Rewrite(input_column="text", llm=model, mode="simplify", target_audience="high school student")
    >> Sink.jsonl("v2_examples/outputs/32_rewrite_simplify.jsonl")
)
pipeline_simplify.run()

# Formalize casual text
pipeline_formalize = (
    Source.list([{"text": "so basically the cells use this chain thing to make energy, pretty cool right?"}])
    >> Rewrite(input_column="text", llm=model, mode="formalize")
    >> Sink.jsonl("v2_examples/outputs/32_rewrite_formalize.jsonl")
)
pipeline_formalize.run()

# Custom instruction
pipeline_custom = (
    source
    >> Rewrite(input_column="text", llm=model, mode="custom", custom_instruction="Rewrite as a haiku")
    >> Sink.jsonl("v2_examples/outputs/32_rewrite_custom.jsonl")
)
pipeline_custom.run()

# Audience-targeted rewrite
pipeline_audience = (
    source
    >> Rewrite(input_column="text", llm=model, mode="audience", target_audience="5-year-old child")
    >> Sink.jsonl("v2_examples/outputs/32_rewrite_audience.jsonl")
)
pipeline_audience.run()

# Elaborate with more detail
pipeline_elaborate = (
    Source.list([{"text": "Gravity pulls things down."}])
    >> Rewrite(input_column="text", llm=model, mode="elaborate")
    >> Sink.jsonl("v2_examples/outputs/32_rewrite_elaborate.jsonl")
)
pipeline_elaborate.run()
