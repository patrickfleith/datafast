"""Source.list and Source.jsonl for loading data; round-trip to Parquet.

Demonstrates: Source.list, Source.jsonl, Sink.jsonl, Sink.parquet
"""

from datafast_v2 import Source, Sink

# --- Part A: Source.list → Sink.jsonl ---
pipeline_a = (
    Source.list([
        {"id": 1, "text": "The sun is a star.", "topic": "astronomy"},
        {"id": 2, "text": "Water boils at 100°C.", "topic": "physics"},
        {"id": 3, "text": "DNA carries genetic info.", "topic": "biology"},
    ])
    >> Sink.jsonl("v2_examples/outputs/05_from_list.jsonl")
)
pipeline_a.run()

# --- Part B: Source.jsonl → Sink.parquet (round-trip) ---
pipeline_b = (
    Source.jsonl("v2_examples/outputs/05_from_list.jsonl")
    >> Sink.parquet("v2_examples/outputs/05_round_trip.parquet")
)
pipeline_b.run()
