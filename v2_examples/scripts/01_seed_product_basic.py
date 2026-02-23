"""Basic Seed.product: cartesian product of value dimensions.

Demonstrates: Seed.values, Seed.product, Sink.jsonl
Output: 3 topics × 2 difficulties × 2 languages = 12 records
"""

from datafast_v2 import Seed, Sink

pipeline = (
    Seed.product(
        Seed.values("topic", ["Physics", "Math", "History"]),
        Seed.values("difficulty", ["easy", "hard"]),
        Seed.values("language", ["en", "fr"]),
    )
    >> Sink.jsonl("v2_examples/outputs/01_seed_product.jsonl")
)

records = pipeline.run()
