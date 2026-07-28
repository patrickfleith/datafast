"""Space text-generation cookbook: seed grid -> technical text corpus.

Demonstrates: Seed.product, LLMStep JSON mode, multi-language generation,
num_outputs, checkpointing, JSONL output, and optional Hub push.

Requires:
- OPENROUTER_API_KEY
- Hugging Face authentication only if DATAFAST_PUSH_TO_HUB=1
- network access to OpenRouter, and to Hugging Face when publishing
"""

from __future__ import annotations

import os

import litellm
from dotenv import load_dotenv

from datafast import AddUUID, LLMStep, Map, Seed, Sink, openrouter

load_dotenv()
litellm.suppress_debug_info = True


SEED = 20250304
MODEL_IDS = ["nvidia/nemotron-3-super-120b-a12b:nitro"]
OUTPUT_PATH = "examples/outputs/44_space_text_generation_cookbook.jsonl"
CHECKPOINT_DIR = "examples/checkpoints/44_space_text_generation_cookbook"
HF_REPO_ID = "patrickfleith/datafast-space-text-generation-cookbook"
NUM_OUTPUTS = 1
PROMPT_PATH = "docs/cookbook/assets/space_text_generation.txt"

DOCUMENT_TYPES = [
    "space engineering textbook",
    "spacecraft design justification document",
    "personal blog of a space engineer",
]

TOPICS = [
    "Microgravity",
    "Vacuum",
    "Heavy Ions",
    "Thermal Extremes",
    "Atomic Oxygen",
    "Debris Impact",
    "Electrostatic Charging",
    "Propellant Boil-off",
]

EXPERTISE_LEVELS = [
    "executives",
    "senior engineers",
    "PhD candidates",
]

LANGUAGES = {
    "en": "English",
    "fr": "French",
}


def make_models():
    return [openrouter(model_id, temperature=0.7) for model_id in MODEL_IDS]


def expected_row_count(model_count: int | None = None) -> int:
    """Return the number of rows this configuration should generate."""
    model_total = len(MODEL_IDS) if model_count is None else model_count
    return (
        len(DOCUMENT_TYPES)
        * len(TOPICS)
        * len(EXPERTISE_LEVELS)
        * len(LANGUAGES)
        * NUM_OUTPUTS
        * model_total
    )


def finalize_record(record: dict) -> dict:
    """Keep the columns meant for publication."""
    return {
        "document_type": record["document_type"],
        "topic": record["topic"],
        "expertise_level": record["expertise_level"],
        "language": record.get("_language", ""),
        "model": record.get("_model", ""),
        "title": record["title"],
        "text": record["text"],
    }


def build_pipeline():
    return (
        Seed.product(
            Seed.values("document_type", DOCUMENT_TYPES),
            Seed.values("topic", TOPICS),
            Seed.values("expertise_level", EXPERTISE_LEVELS),
        ).as_step("seed_space_text_grid")
        >> LLMStep(
            prompt=PROMPT_PATH,
            input_columns=["document_type", "topic", "expertise_level"],
            output_columns=["title", "text"],
            parse_mode="json",
            model=make_models(),
            language=LANGUAGES,
            num_outputs=NUM_OUTPUTS,
            on_parse_error="raise",
        ).as_step("generate_space_text")
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
            commit_message=f"Publish cookbook 44 text dataset with {', '.join(MODEL_IDS)}",
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
