"""Classify: multi-label (LLM) and function-based classification.

Demonstrates:
  Part A — multi_label=True with labels_description (LLM)
  Part B — fn-based classification (no LLM needed)
"""

from datafast_v2 import Source, Classify, Sink, openrouter

# --- Part A: Multi-label LLM classification ---
# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.3)

pipeline_llm = (
    Source.list([
        {"article": "New AI chip beats benchmarks while government debates regulation."},
        {"article": "The Lakers won the championship in a thrilling overtime game."},
        {"article": "Tech stocks surge as Congress passes data privacy bill."},
    ])
    >> Classify(
        labels=["politics", "sports", "technology", "entertainment"],
        input_columns=["article"],
        output_column="topics",
        multi_label=True,
        llm=model,
        labels_description={
            "politics": "Government, elections, policy, regulation",
            "sports": "Athletic events, teams, scores, players",
            "technology": "Tech industry, software, hardware, AI",
            "entertainment": "Movies, music, TV, celebrity news",
        },
    )
    >> Sink.jsonl("v2_examples/outputs/28_classify_multilabel.jsonl")
)
pipeline_llm.run()

# --- Part B: Function-based classification (no LLM) ---
pipeline_fn = (
    Source.list([
        {"text": "Short."},
        {"text": "A medium-length sentence with a few words in it."},
        {"text": "A much longer piece of text that goes on and on with many words and details about various topics."},
    ])
    >> Classify(
        labels=["short", "medium", "long"],
        input_columns=["text"],
        output_column="length_class",
        fn=lambda r: "short" if len(r["text"]) < 20 else ("medium" if len(r["text"]) < 80 else "long"),
    )
    >> Sink.list()
)
pipeline_fn.run()
