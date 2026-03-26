"""Join step: merge two datasets by key (inner, left, right, outer).

Demonstrates: Join on a shared key column with different join types.
"""

from loguru import logger

from datafast import Source, Join, Sink

users = Source.list([
    {"user_id": 1, "name": "Alice"},
    {"user_id": 2, "name": "Bob"},
    {"user_id": 3, "name": "Charlie"},
    {"user_id": 4, "name": "Diana"},
])

actions = Source.list([
    {"user_id": 1, "action": "login", "timestamp": "2025-01-01"},
    {"user_id": 1, "action": "purchase", "timestamp": "2025-01-02"},
    {"user_id": 2, "action": "login", "timestamp": "2025-01-01"},
    {"user_id": 5, "action": "login", "timestamp": "2025-01-03"},
])

# Inner join: only matching user_ids (1 and 2)
pipeline_inner = users >> Join(actions, on="user_id") >> Sink.list()
r_inner = pipeline_inner.run()
logger.info(f"Inner join: {len(r_inner)} records")
for r in r_inner:
    logger.info(f"  {r['name']} — {r['action']} at {r['timestamp']}")

# Left join: all users, even without actions (how="left")
pipeline_left = users >> Join(actions, on="user_id", how="left") >> Sink.list()
r_left = pipeline_left.run()
logger.info(f"Left join: {len(r_left)} records")
