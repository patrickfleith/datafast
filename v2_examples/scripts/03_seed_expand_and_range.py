"""Seed.expand (hierarchical) and Seed.range (numeric) combined via product.

Demonstrates: Seed.expand, Seed.range, Seed.product, Sink.jsonl
Output: 6 domain/topic pairs × 5 grades = 30 records
"""

from datafast_v2 import Seed, Sink

pipeline = (
    Seed.product(
        Seed.expand("domain", "topic", {
            "Science": ["Chemistry", "Physics", "Biology"],
            "Math": ["Algebra", "Calculus", "Statistics"],
        }),
        Seed.range("grade", 1, 5),
    )
    >> Sink.jsonl("v2_examples/outputs/03_seed_expand_range.jsonl")
)

records = pipeline.run()
