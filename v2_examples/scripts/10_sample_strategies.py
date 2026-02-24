"""Sample step with various strategies.

Demonstrates: top, bottom, weighted, gaussian, first, last, systematic, stratified.
"""

from loguru import logger

from datafast_v2 import Source, Sample, Sink

data = [
    {"id": i, "score": i * 3.5, "category": ["A", "B", "C"][i % 3]}
    for i in range(1, 31)
]
source = Source.list(data)

# Top N by score (highest scores)
r_top = (source >> Sample(n=5, strategy="top", by="score") >> Sink.list()).run()
logger.info(f"top(5, by=score): scores={[r['score'] for r in r_top]}")

# Bottom N by score (lowest scores)
r_bot = (source >> Sample(n=5, strategy="bottom", by="score") >> Sink.list()).run()
logger.info(f"bottom(5, by=score): scores={[r['score'] for r in r_bot]}")

# Weighted random (higher score → higher chance of selection)
r_wt = (source >> Sample(n=5, strategy="weighted", by="score", seed=42) >> Sink.list()).run()
logger.info(f"weighted(5, by=score): ids={[r['id'] for r in r_wt]}")

# Gaussian around a center value
r_gauss = (source >> Sample(n=5, strategy="gaussian", by="score", center=50.0, std=10.0, seed=42) >> Sink.list()).run()
logger.info(f"gaussian(5, center=50, std=10): scores={[r['score'] for r in r_gauss]}")

# Stratified by category (maintains proportional distribution)
r_strat = (source >> Sample(n=9, strategy="stratified", by="category", seed=42) >> Sink.list()).run()
cats = {}
for r in r_strat:
    cats[r["category"]] = cats.get(r["category"], 0) + 1
logger.info(f"stratified(9, by=category): distribution={cats}")

# Systematic: every Nth record
r_sys = (source >> Sample(strategy="systematic", step=5) >> Sink.list()).run()
logger.info(f"systematic(step=5): ids={[r['id'] for r in r_sys]}")
