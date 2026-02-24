"""Seed.zip: element-wise pairing of same-length dimensions.

Demonstrates: Seed.zip, Seed.values, Sink.csv
Output: 5 paired records (city ↔ country ↔ population)
"""

from datafast_v2 import Seed, Sink

pipeline = (
    Seed.zip(
        Seed.values("city", ["Paris", "Berlin", "Madrid", "Rome", "London"]),
        Seed.values("country", ["France", "Germany", "Spain", "Italy", "UK"]),
        Seed.values("population_millions", [2.1, 3.6, 3.2, 2.8, 8.9]),
    )
    >> Sink.csv("v2_examples/outputs/02_seed_zip.csv")
)

records = pipeline.run()
