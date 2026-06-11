"""Text-classification cookbook: seed grid -> multilingual trail reports.

Demonstrates: Seed.product, prompt expansion via seed dimensions, multi-model
generation, multi-language generation, checkpointing, JSONL output, and
optional Hugging Face Hub publishing.

Requires:
- OPENROUTER_API_KEY
- Hugging Face authentication only if DATAFAST_PUSH_TO_HUB=1
- network access to OpenRouter, and to Hugging Face when publishing
"""

from __future__ import annotations

import os

import litellm
from dotenv import load_dotenv

from datafast import AddUUID, LLMStep, Map, Seed, SeedDimension, Sink, openrouter

load_dotenv()
litellm.suppress_debug_info = True


SEED = 20250611
MODEL_IDS = [
    "nvidia/nemotron-3-super-120b-a12b:nitro",
    "mistralai/ministral-14b-2512",
]
OUTPUT_PATH = "examples/outputs/45_text_classification_cookbook.jsonl"
CHECKPOINT_DIR = "examples/checkpoints/45_text_classification_cookbook"
HF_REPO_ID = "patrickfleith/datafast-text-classification-cookbook"
PROMPT_PATH = "docs/cookbook/assets/text_classification_generation.txt"

LABELS = [
    {
        "label_name": "trail_obstruction",
        "label_description": (
            "The trail is partially or fully blocked by obstacles such as "
            "fallen trees, landslides, snow, flooding, erosion, or dense "
            "vegetation."
        ),
    },
    {
        "label_name": "infrastructure_issues",
        "label_description": (
            "The report is about damaged or missing bridges, signs, stairs, "
            "handrails, markers, boardwalks, or similar trail infrastructure."
        ),
    },
    {
        "label_name": "hazards",
        "label_description": (
            "The trail has immediate safety risks such as slippery surfaces, "
            "dangerous crossings, unstable terrain, wildlife threats, or "
            "other hazardous conditions."
        ),
    },
    {
        "label_name": "positive_conditions",
        "label_description": (
            "The report highlights clear, safe, enjoyable trail conditions "
            "such as good maintenance, solid infrastructure, clear signage, "
            "or scenic features."
        ),
    },
]

TRAIL_TYPES = [
    "mountain trail",
    "coastal path",
    "forest walk",
]

STYLES = [
    "a brief social media post",
    "a hiking review",
]

LANGUAGES = {
    "en": "English",
    "fr": "French",
}


def make_models():
    return [openrouter(model_id, temperature=0.8) for model_id in MODEL_IDS]


def make_label_dimension() -> SeedDimension:
    return SeedDimension(
        columns=["label_name", "label_description"],
        values=LABELS,
    )


def expected_row_count(model_count: int | None = None) -> int:
    """Return the number of rows this configuration should generate."""
    model_total = len(MODEL_IDS) if model_count is None else model_count
    return (
        len(LABELS)
        * len(TRAIL_TYPES)
        * len(STYLES)
        * len(LANGUAGES)
        * model_total
    )


def finalize_record(record: dict) -> dict:
    """Keep the publication fields and flatten generation metadata."""
    return {
        "label": record["label_name"],
        "label_description": record["label_description"],
        "label_source": "synthetic",
        "trail_type": record["trail_type"],
        "style": record["style"],
        "language": record.get("_language", ""),
        "model": record.get("_model", ""),
        "text": record["text"],
    }


def build_pipeline():
    return (
        Seed.product(
            make_label_dimension(),
            Seed.values("trail_type", TRAIL_TYPES),
            Seed.values("style", STYLES),
        ).as_step("seed_trail_report_grid")
        >> LLMStep(
            prompt=PROMPT_PATH,
            input_columns=["label_name", "label_description", "trail_type", "style"],
            output_column="text",
            parse_mode="text",
            model=make_models(),
            language=LANGUAGES,
        ).as_step("generate_trail_reports")
        >> Map(finalize_record).as_step("finalize_record")
        >> AddUUID(column="id", overwrite=True).as_step("add_uuid")
        >> Sink.jsonl(OUTPUT_PATH)
    )


def push_records_to_hub(records: list[dict]) -> None:
    list(
        Sink.hub(
            repo_id=HF_REPO_ID,
            private=True,
            train_size=0.8,
            seed=SEED,
            shuffle=True,
            commit_message=(
                "Publish cookbook 45 classification dataset with "
                f"{', '.join(MODEL_IDS)}"
            ),
        ).process(records)
    )


def main() -> None:
    print(f"Expected rows: {expected_row_count()}")
    records = build_pipeline().run(
        batch_size=4,
        checkpoint_dir=CHECKPOINT_DIR,
        resume=True,
    )

    if os.getenv("DATAFAST_PUSH_TO_HUB") == "1":
        push_records_to_hub(records)

    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
