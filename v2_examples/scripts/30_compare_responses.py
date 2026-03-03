"""Compare step: pairwise comparison of two response columns.

Demonstrates: Compare with LLM (detailed output_mode) and function-based.
"""

from datafast_v2 import Source, Compare, Sink, openrouter
import litellm
litellm.suppress_debug_info = True

# --- LLM-based comparison with detailed output ---
# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.3)

pipeline_llm = (
    Source.list([
        {
            "question": "What causes rain?",
            "response_a": "Rain is caused by the water cycle: evaporation, condensation into clouds, and precipitation when droplets become heavy enough.",
            "response_b": "Water falls from clouds.",
        },
        {
            "question": "How does a computer work?",
            "response_a": "A computer processes binary instructions through its CPU, stores data in memory, and uses I/O devices for interaction.",
            "response_b": "A computer uses a processor to execute instructions stored in memory, coordinating with storage, RAM, and peripherals via a bus system.",
        },
    ])
    >> Compare(
        column_a="response_a",
        column_b="response_b",
        criteria="helpfulness, accuracy, and completeness",
        output_mode="detailed",
        llm=model,
    )
    >> Sink.jsonl("v2_examples/outputs/30_compare_llm.jsonl")
)
pipeline_llm.run()

# --- Function-based comparison (by length) ---
pipeline_fn = (
    Source.list([
        {"summary_a": "Short version.", "summary_b": "A much longer and more detailed version of the summary."},
        {"summary_a": "A comprehensive explanation.", "summary_b": "Brief."},
    ])
    >> Compare(
        column_a="summary_a",
        column_b="summary_b",
        criteria="length",
        fn=lambda r: "a" if len(r["summary_a"]) > len(r["summary_b"]) else "b",
    )
    >> Sink.list()
)
pipeline_fn.run()
