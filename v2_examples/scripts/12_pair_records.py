"""Pair step: create pairs (or n-tuples) of records.

Demonstrates: Pair with random, sliding, sequential strategies, within constraint.
"""

from loguru import logger

from datafast_v2 import Source, Pair, Sink

data = [
    {"doc_id": 1, "text": "Intro to algebra", "category": "math"},
    {"doc_id": 2, "text": "Advanced calculus", "category": "math"},
    {"doc_id": 3, "text": "Quantum physics", "category": "science"},
    {"doc_id": 4, "text": "Organic chemistry", "category": "science"},
    {"doc_id": 5, "text": "Linear algebra", "category": "math"},
    {"doc_id": 6, "text": "Thermodynamics", "category": "science"},
]
source = Source.list(data)

# Random pairs (default n=2)
r1 = (source >> Pair(n=2, strategy="random", max_pairs=4, seed=42) >> Sink.list()).run()
logger.info(f"Random pairs: {len(r1)} pairs")

# Sliding window pairs
r2 = (source >> Pair(n=2, strategy="sliding") >> Sink.list()).run()
print(r2)

# logger.info(f"Sliding pairs: {len(r2)} pairs")

# Pairs within the same category
r3 = (source >> Pair(n=2, strategy="random", within="category", max_pairs=6, seed=42) >> Sink.list()).run()
for r in r3:
    logger.info(
        "  Within-category pair: "
        f"{r.get('chunk_1_category')} — doc {r.get('chunk_1_doc_id')} & {r.get('chunk_2_doc_id')}"
    )

# List output format (items as a list instead of separate columns)
r4 = (source >> Pair(n=2, strategy="sequential", output_format="list") >> Sink.list()).run()
logger.info(
    f"Sequential pairs (list format): {len(r4)} pairs, "
    f"first has {len(r4[0]['chunks'])} items"
)
