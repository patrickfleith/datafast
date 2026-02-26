"""Group step: aggregate records by key columns.

Demonstrates: Group with collect, agg functions, and min_per_group.
"""

from loguru import logger

from datafast_v2 import Source, Group, Sink

data = [
    {"product_id": "A", "rating": 5, "review": "Excellent product!"},
    {"product_id": "A", "rating": 4, "review": "Very good quality"},
    {"product_id": "A", "rating": 3, "review": "Decent, nothing special"},
    {"product_id": "B", "rating": 2, "review": "Poor durability"},
    {"product_id": "B", "rating": 4, "review": "Surprisingly good"},
    {"product_id": "C", "rating": 5, "review": "Best purchase ever"},
]
source = Source.list(data)

# Collect all reviews per product
r1 = (source >> Group(by="product_id", collect="review") >> Sink.list()).run()
for r in r1:
    logger.info(f"Product {r['product_id']}: {len(r['review'])} reviews collected")

# Aggregations: mean rating and count per product
r2 = (
    source
    >> Group(by="product_id", agg={"avg_rating": "rating:mean", "num_reviews": "rating:count"})
    >> Sink.list()
).run()
for r in r2:
    logger.info(f"Product {r['product_id']}: avg={r['avg_rating']:.1f}, count={r['num_reviews']}")

# Filter out products with fewer than 2 reviews
r3 = (
    source
    >> Group(by="product_id", collect="review", min_per_group=2)
    >> Sink.list()
).run()
logger.info(f"Products with >= 2 reviews: {[r['product_id'] for r in r3]}")
