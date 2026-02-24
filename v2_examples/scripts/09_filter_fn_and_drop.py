"""Filter with custom function and keep=False to drop matches.

Demonstrates: Filter(fn=...) for complex conditions, Filter(keep=False) to invert.
"""

from datafast_v2 import Source, Filter, Sink

data = [
    {"id": 1, "text": "A detailed explanation of quantum mechanics", "quality": 8},
    {"id": 2, "text": "Ok", "quality": 2},
    {"id": 3, "text": "Comprehensive guide to neural network architectures", "quality": 9},
    {"id": 4, "text": "Meh", "quality": 1},
    {"id": 5, "text": "Introduction to linear algebra for beginners", "quality": 7},
]

pipeline = (
    Source.list(data)
    # Keep only records with long text AND high quality
    >> Filter(fn=lambda r: len(r["text"]) > 20 and r["quality"] >= 7)
    # Drop records that contain "neural" (keep=False inverts the match)
    >> Filter(where={"text": {"$contains": "neural"}}, keep=False)
    >> Sink.list()
)

records = pipeline.run()
