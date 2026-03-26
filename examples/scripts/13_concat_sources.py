"""Concat: stack multiple data sources vertically.

Demonstrates: Concat to combine records from different origins.
"""

from loguru import logger

from datafast import Source, Map, Concat, Sink

source_a = (
    Source.list([
        {"text": "The Earth orbits the Sun", "topic": "astronomy"},
        {"text": "Water is H2O", "topic": "chemistry"},
    ])
    >> Map(lambda r: {**r, "origin": "science_facts"})
)

source_b = (
    Source.list([
        {"text": "Rome was founded in 753 BC", "topic": "history"},
        {"text": "The Magna Carta was signed in 1215", "topic": "history"},
    ])
    >> Map(lambda r: {**r, "origin": "history_facts"})
)

pipeline = Concat(source_a, source_b) >> Sink.jsonl("examples/outputs/13_concat.jsonl")

records = pipeline.run()
logger.info(f"Concatenated {len(records)} records from 2 sources")
