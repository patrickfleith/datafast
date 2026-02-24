"""Declarative Filter with where operators (MongoDB-style).

Demonstrates: $gte, $in, $contains, $len_gt, $or, $and operators.
"""

from loguru import logger

from datafast_v2 import Source, Filter, Sink

data = [
    {"id": 1, "text": "Short text", "score": 9, "category": "science"},
    {"id": 2, "text": "A moderately long piece of text about technology and AI", "score": 6, "category": "tech"},
    {"id": 3, "text": "Brief", "score": 3, "category": "science"},
    {"id": 4, "text": "An in-depth article about machine learning models", "score": 8, "category": "tech"},
    {"id": 5, "text": "History of ancient civilizations and their impact", "score": 7, "category": "history"},
    {"id": 6, "text": "Quick note", "score": 2, "category": "misc"},
]
source = Source.list(data)

# Numeric comparison: score >= 7
r1 = (source >> Filter(where={"score": {"$gte": 7}}) >> Sink.list()).run()

# Set membership: category in [science, tech]
r2 = (source >> Filter(where={"category": {"$in": ["science", "tech"]}}) >> Sink.list()).run()

# String length: text longer than 20 chars
r3 = (source >> Filter(where={"text": {"$len_gt": 20}}) >> Sink.list()).run()

# Logical OR: high score OR science category
r4 = (source >> Filter(where={"$or": [{"score": {"$gte": 9}}, {"category": "science"}]}) >> Sink.list()).run()

# Exact match: category == "tech"
r5 = (source >> Filter(where={"category": "tech"}) >> Sink.list()).run()

logger.info(f"score >= 7:        {len(r1)} records (ids: {[r['id'] for r in r1]})")
logger.info(f"category in [s,t]: {len(r2)} records (ids: {[r['id'] for r in r2]})")
logger.info(f"text len > 20:     {len(r3)} records (ids: {[r['id'] for r in r3]})")
logger.info(f"score>=9 OR sci:   {len(r4)} records (ids: {[r['id'] for r in r4]})")
logger.info(f"category == tech:  {len(r5)} records (ids: {[r['id'] for r in r5]})")
