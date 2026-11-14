"""Text-classification cookbook: seed grid -> multilingual trail comments.

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
        "label": "trail_obstruction",
        "label_description": (
            "The trail is partially or fully blocked by obstacles such as "
            "fallen trees, landslides, snow, flooding, erosion, or dense "
            "vegetation."
        ),
    },
    {
        "label": "infrastructure_issues",
        "label_description": (
            "The report is about damaged or missing bridges, signs, stairs, "
            "handrails, markers, boardwalks, or similar trail infrastructure."
        ),
    },
    {
        "label": "hazards",
        "label_description": (
            "The trail has immediate safety risks such as slippery surfaces, "
            "dangerous crossings, unstable terrain, wildlife threats, or "
            "other hazardous conditions."
        ),
    },
    {
        "label": "positive_conditions",
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

MODELS = [openrouter(model_id, temperature=0.8) for model_id in MODEL_IDS]
EXPECTED_ROWS = (
    len(LABELS)
    * len(TRAIL_TYPES)
    * len(STYLES)
    * len(LANGUAGES)
    * len(MODELS)
)


def keep_output_fields(record: dict) -> dict:
    """Keep only the fields meant for publication."""
    return {
        "label": record["label"],
        "trail_type": record["trail_type"],
        "style": record["style"],
        "language": record.get("_language", ""),
        "model": record.get("_model", ""),
        "text": record["text"],
    }


pipeline = (
    Seed.product(
        SeedDimension(
            columns=["label", "label_description"],
            values=LABELS,
        ),
        Seed.values("trail_type", TRAIL_TYPES),
        Seed.values("style", STYLES),
    ).as_step("seed_trail_report_grid")
    >> LLMStep(
        prompt=PROMPT_PATH,
        input_columns=["label", "label_description", "trail_type", "style"],
        output_column="text",
        parse_mode="text",
        model=MODELS,
        language=LANGUAGES,
    ).as_step("generate_trail_reports")
    >> Map(keep_output_fields).as_step("keep_output_fields")
    >> AddUUID(column="id", overwrite=True).as_step("add_uuid")
    >> Sink.jsonl(OUTPUT_PATH)
)


def main() -> None:
    print(f"Expected rows: {EXPECTED_ROWS}")
    records = pipeline.run(
        batch_size=4,
        checkpoint_dir=CHECKPOINT_DIR,
        resume=True,
    )

    if os.getenv("DATAFAST_PUSH_TO_HUB") == "1":
        list(
            Sink.hub(
                repo_id=HF_REPO_ID,
                private=False,
                train_size=0.8,
                seed=SEED,
                shuffle=True,
                commit_message=(
                    "Publish cookbook 45 classification dataset with "
                    f"{', '.join(MODEL_IDS)}"
                ),
            ).process(records)
        )

    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
