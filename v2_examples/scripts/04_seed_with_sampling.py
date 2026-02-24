"""Large seed product reduced with Sample step.

Demonstrates: Seed.product with many dimensions, Sample (uniform + stratified)
Output: 192 total combinations → 20 sampled records
"""

from datafast_v2 import Seed, Sample, Sink

personas = Seed.values("persona", ["student", "teacher", "researcher", "professional"])
topics = Seed.expand("topic", "subtopic", {
    "Physics": ["Quantum Mechanics", "Thermodynamics", "Relativity"],
    "Biology": ["Genetics", "Ecology", "Neuroscience"],
})
languages = Seed.values("language", ["en", "fr", "de", "es"])
difficulty = Seed.range("difficulty", 1, 4)

full_product = Seed.product(personas, topics, languages, difficulty)

pipeline = (
    full_product
    >> Sample(n=20, strategy="uniform", seed=42)
    # Alternative: stratified sampling to maintain persona distribution
    # >> Sample(n=20, strategy="stratified", by="persona", seed=42)
    >> Sink.jsonl("v2_examples/outputs/04_seed_sampled.jsonl")
)

records = pipeline.run()
