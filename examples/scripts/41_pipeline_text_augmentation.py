"""Text augmentation pipeline: original data + paraphrased versions via Concat.

Demonstrates: Rewrite for augmentation, Concat to merge original + rewritten data.
"""

from datafast import Source, Map, Rewrite, Concat, Sink, openrouter

import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.8)

original = Source.list([
    {"text": "Machine learning models learn from data.", "label": "tech"},
    {"text": "The heart pumps blood through the body.", "label": "biology"},
    {"text": "Democracy requires citizen participation.", "label": "politics"},
])

# Tag original records
tagged_original = original >> Map(lambda r: {**r, "is_augmented": False})

# Create paraphrased augmentations
augmented = (
    original
    >> Rewrite(
        input_column="text",
        llm=model,
        mode="paraphrase",
        num_variations=2,
    )
    >> Map(lambda r: {**r, "is_augmented": True})
)

# Combine original + augmented
pipeline = (
    Concat(tagged_original, augmented)
    >> Sink.jsonl("examples/outputs/41_text_augmentation.jsonl")
)

records = pipeline.run()
