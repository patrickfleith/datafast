"""Map step: apply a function to each record one-to-one.

Demonstrates: Map to add computed fields and transform existing ones.
"""

from datafast_v2 import Source, Map, Sink

pipeline = (
    Source.list([
        {"id": 1, "text": "The quick brown fox jumps over the lazy dog"},
        {"id": 2, "text": "Machine learning is a subset of AI"},
        {"id": 3, "text": "Python is great for data science and web development"},
    ])
    >> Map(lambda r: {**r, "word_count": len(r["text"].split())})
    >> Map(lambda r: {**r, "text_upper": r["text"].upper()})
    >> Sink.jsonl("v2_examples/outputs/06_map_transform.jsonl")
)

records = pipeline.run()
