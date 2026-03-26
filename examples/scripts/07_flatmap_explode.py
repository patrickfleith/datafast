"""FlatMap: transform each record into zero or more records.

Demonstrates: FlatMap to explode list fields and cross-join with new dimensions.
"""

from datafast import Source, FlatMap, Sink

# --- Explode a list field: one record per tag ---
pipeline_explode = (
    Source.list([
        {"id": 1, "text": "Neural networks overview", "tags": ["AI", "deep learning"]},
        {"id": 2, "text": "Sorting algorithms", "tags": ["CS", "algorithms", "data structures"]},
        {"id": 3, "text": "Climate change report", "tags": ["science"]},
    ])
    >> FlatMap(lambda r: [
        {"id": r["id"], "text": r["text"], "tag": tag}
        for tag in r["tags"]
    ])
    >> Sink.jsonl("examples/outputs/07_flatmap_explode.jsonl")
)
records_explode = pipeline_explode.run()

# --- Cross-join with new dimensions: duplicate each record with style variants ---
pipeline_cross = (
    Source.list([
        {"topic": "Physics", "question": "What is gravity?"},
        {"topic": "History", "question": "Who built the pyramids?"},
    ])
    >> FlatMap(lambda r: [
        {**r, "style": style}
        for style in ["formal", "conversational", "academic"]
    ])
    >> Sink.jsonl("examples/outputs/07_flatmap_cross.jsonl")
)
records_cross = pipeline_cross.run()
