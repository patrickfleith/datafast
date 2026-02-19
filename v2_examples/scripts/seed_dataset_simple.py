from datafast_v2.sources.seed import Seed
from datafast_v2.transforms.sample import Sample
from datafast_v2.sinks.sink import Sink

pipeline = (
    Seed.product(
        Seed.values("topic", ["Physics", "Math", "History"]),
        Seed.values("difficulty", ["easy", "medium", "hard"]),
        Seed.values("language", ["en", "fr"]),
    )
    >> Sample(n=10, strategy="uniform", seed=42)
    >> Sink.jsonl("v2_examples/outputs/sampled_seeds.jsonl")
)

records = pipeline.run()  # 10 sampled records, zero LLM calls