"""Showcase of all Seed and Sample features for creating datasets without LLM calls.

Demonstrates:
  Seed dimensions:
    - Seed.values()  — explicit list of values
    - Seed.expand()  — parent-child hierarchical mapping
    - Seed.range()   — numeric ranges (inclusive)

  Seed combinators:
    - Seed.product() — cartesian product (all combinations)
    - Seed.zip()     — element-wise pairing (same-length dimensions)

  Sampling strategies (via Sample step):
    - uniform     — random selection
    - first/last  — take from start or end
    - systematic  — every Nth record
    - stratified  — maintain category distribution

  Sinks:
    - Sink.jsonl() — save to JSONL file
    - Sink.csv()   — save to CSV file
"""

from loguru import logger

from datafast_v2.sources.seed import Seed
from datafast_v2.sinks.sink import Sink
from datafast_v2.transforms.sample import Sample

# ---------------------------------------------------------------------------
# 1. Seed.values — explicit lists
# ---------------------------------------------------------------------------
personas = Seed.values("persona", ["student", "teacher", "researcher", "professional"])
languages = Seed.values("language", ["en", "fr", "de", "es"])

# ---------------------------------------------------------------------------
# 2. Seed.expand — hierarchical parent → child mapping
# ---------------------------------------------------------------------------
topics = Seed.expand("topic", "subtopic", {
    "Physics": ["Quantum Mechanics", "Thermodynamics", "Relativity"],
    "Biology": ["Genetics", "Ecology", "Neuroscience"],
    "History": ["Ancient Rome", "World War II", "Industrial Revolution"],
    "Computer Science": ["Algorithms", "Machine Learning", "Databases"],
})

# ---------------------------------------------------------------------------
# 3. Seed.range — numeric ranges (inclusive on both ends)
# ---------------------------------------------------------------------------
grade_levels = Seed.range("grade_level", 1, 12)
difficulty = Seed.range("difficulty_score", 1, 5)

# ---------------------------------------------------------------------------
# 4. Seed.product — cartesian product of all dimensions
#    Total: 4 personas × 12 topic/subtopic pairs × 4 languages × 5 difficulty
#         = 960 records
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("EXAMPLE 1: Seed.product — full cartesian product + sampling")
logger.info("=" * 60)

full_product = Seed.product(personas, topics, languages, difficulty)

pipeline_product = (
    full_product
    >> Sample(n=50, strategy="uniform", seed=42)
    >> Sink.jsonl("v2_examples/outputs/showcase_product_sampled.jsonl")
)
records_product = pipeline_product.run()
logger.info(f"Product pipeline: {len(full_product)} total combinations → {len(records_product)} sampled\n")

# ---------------------------------------------------------------------------
# 5. Seed.zip — element-wise pairing (dimensions must have same length)
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("EXAMPLE 2: Seed.zip — paired dimensions")
logger.info("=" * 60)

pipeline_zip = (
    Seed.zip(
        Seed.values("city", ["Paris", "Berlin", "Madrid", "Rome", "London"]),
        Seed.values("country", ["France", "Germany", "Spain", "Italy", "UK"]),
        Seed.values("population_millions", [2.1, 3.6, 3.2, 2.8, 8.9]),
    )
    >> Sink.csv("v2_examples/outputs/showcase_zip.csv")
)
records_zip = pipeline_zip.run()
logger.info(f"Zip pipeline: {len(records_zip)} paired records\n")

# ---------------------------------------------------------------------------
# 6. Mixing product and zip — product of a zipped dimension with others
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("EXAMPLE 3: Combining expand + values + range in a product")
logger.info("=" * 60)

pipeline_mixed = (
    Seed.product(
        Seed.expand("subject", "concept", {
            "Math": ["Algebra", "Calculus", "Statistics"],
            "Science": ["Chemistry", "Physics", "Biology"],
        }),
        Seed.values("question_type", ["multiple_choice", "open_ended", "true_false"]),
        Seed.range("complexity", 1, 3),
    )
    >> Sink.jsonl("v2_examples/outputs/showcase_mixed.jsonl")
)
records_mixed = pipeline_mixed.run()
logger.info(f"Mixed pipeline: {len(records_mixed)} records (no sampling, all combos)\n")

# ---------------------------------------------------------------------------
# 7. Sampling strategies showcase
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("EXAMPLE 4: Sampling strategies")
logger.info("=" * 60)

# Create a base seed with a stratifiable column
base = Seed.product(
    Seed.values("category", ["A", "B", "C"]),
    Seed.range("id", 1, 30),
)

# 7a. First N
pipeline_first = (
    base
    >> Sample(n=5, strategy="first")
    >> Sink.jsonl("v2_examples/outputs/showcase_sample_first.jsonl")
)
records_first = pipeline_first.run()
logger.info(f"  first(5): {len(records_first)} records")

# 7b. Last N
pipeline_last = (
    base
    >> Sample(n=5, strategy="last")
    >> Sink.jsonl("v2_examples/outputs/showcase_sample_last.jsonl")
)
records_last = pipeline_last.run()
logger.info(f"  last(5):  {len(records_last)} records")

# 7c. Systematic — every 10th record
pipeline_systematic = (
    base
    >> Sample(strategy="systematic", step=10)
    >> Sink.jsonl("v2_examples/outputs/showcase_sample_systematic.jsonl")
)
records_systematic = pipeline_systematic.run()
logger.info(f"  systematic(step=10): {len(records_systematic)} records")

# 7d. Stratified by category — maintains proportions
pipeline_stratified = (
    base
    >> Sample(n=15, strategy="stratified", by="category", seed=42)
    >> Sink.jsonl("v2_examples/outputs/showcase_sample_stratified.jsonl")
)
records_stratified = pipeline_stratified.run()
logger.info(f"  stratified(15, by=category): {len(records_stratified)} records")

# 7e. Fraction-based sampling
pipeline_frac = (
    base
    >> Sample(frac=0.25, strategy="uniform", seed=42)
    >> Sink.jsonl("v2_examples/outputs/showcase_sample_fraction.jsonl")
)
records_frac = pipeline_frac.run()
logger.info(f"  uniform(frac=0.25): {len(records_frac)} records from {len(base)}\n")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("ALL OUTPUTS SAVED to v2_examples/outputs/")
logger.info("=" * 60)
