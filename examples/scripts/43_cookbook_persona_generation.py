"""Persona-generation cookbook: XSum article -> personas -> user prompts.

Demonstrates: Source.huggingface, Map, Filter, Sample, JSON-mode LLMSteps,
and prompt assets stored under docs/cookbook/assets.

Requires:
- MISTRAL_API_KEY
- PERSONA_COOKBOOK_HF_REPO_ID
- Hugging Face authentication via HF_TOKEN or a cached `huggingface_hub` login
- network access to Hugging Face and Mistral AI
"""

import random

from dotenv import load_dotenv

from datafast import Filter, LLMStep, Map, Sample, Sink, Source, openrouter

import litellm

load_dotenv()

litellm.suppress_debug_info = True


MODEL_ID = "nvidia/nemotron-3-super-120b-a12b:nitro"
OUTPUT_PATH = "examples/outputs/43_persona_cookbook.jsonl"
HF_REPO_ID = "patrickfleith/new-persona-cookbook-dataset"
TEXT_TO_PERSONA_PROMPTS = [
    "docs/cookbook/assets/text_to_persona_v1.txt",
    "docs/cookbook/assets/text_to_persona_v2.txt",
    "docs/cookbook/assets/text_to_persona_v3.txt",
]
PERSONA_TO_PERSONA_PROMPTS = [
    "docs/cookbook/assets/persona_to_persona_v1.txt",
    "docs/cookbook/assets/persona_to_persona_v2.txt",
    "docs/cookbook/assets/persona_to_persona_v3.txt",
]
# PERSONA_TO_USER_PROMPTS = [
#     "docs/cookbook/assets/persona_to_user_prompt_v2.txt",
#     "docs/cookbook/assets/persona_to_user_prompt_v3.txt",
# ]
LIFE_STAGES = [
    "a teenager",
    "a young adult",
    "an adult (30s/40s)",
    "a middle-aged person (in their 50s/60s)",
    "a senior person (in their 70s/80s)",
]


def add_word_count(record: dict) -> dict:
    return {**record, "word_count": len(record["document"].split())}


def assign_life_stage(record: dict) -> dict:
    return {**record, "life_stage": random.choice(LIFE_STAGES)}


def assign_related_life_stage(record: dict) -> dict:
    return {**record, "related_life_stage": random.choice(LIFE_STAGES)}


def keep_output_fields(record: dict) -> dict:
    return {
        "summary": record["summary"],
        "document": record["document"],
        "word_count": record["word_count"],
        "life_stage": record["life_stage"],
        "persona_description": record["persona_description"],
        "relationship_type": record["relationship_type"],
        "related_life_stage": record["related_life_stage"],
        "related_persona_description": record["related_persona_description"],
    }


def build_pipeline():
    model = openrouter(MODEL_ID, temperature=0.7)

    return (
        Source.huggingface(
            "xsum",
            split="validation",
            columns=["document", "summary"],
        )
    >> Map(add_word_count).as_step("add_word_count")
    >> Filter(fn=lambda r: 300 <= r["word_count"] <= 500).as_step("filter_word_count")
    >> Sample(n=100, strategy="first").as_step("take_first_100")
    >> Map(assign_life_stage).as_step("assign_life_stage")
    >> LLMStep(
        prompt=Sample(TEXT_TO_PERSONA_PROMPTS, n=1),
        input_columns=["document", "life_stage"],
        output_columns=["persona_description"],
        model=model,
        parse_mode="json",
        on_parse_error="raise",
    ).as_step("text_to_persona")
    >> Map(assign_related_life_stage).as_step("assign_related_life_stage")
    >> LLMStep(
        prompt=Sample(PERSONA_TO_PERSONA_PROMPTS, n=1),
        input_columns=["persona_description", "related_life_stage"],
        output_columns=["relationship_type", "related_persona_description"],
        model=model,
        parse_mode="json",
        on_parse_error="raise",
    ).as_step("persona_to_persona")
    >> Map(keep_output_fields).as_step("keep_output_fields")
    >> Sink.jsonl(OUTPUT_PATH)
    >> Sink.hub(HF_REPO_ID, private=True)
)


def push_records_to_hub(records: list[dict]) -> None:
    repo_id = "patrickfleith/datafast-persona-cookbook"
    private = False

    list(
        Sink.hub(
            repo_id=repo_id,
            private=private,
            commit_message=f"Publish cookbook 43 persona dataset with {MODEL_ID}",
        ).process(records)
    )


def main() -> None:
    records = build_pipeline().run(batch_size=1)
    push_records_to_hub(records)


if __name__ == "__main__":
    main()
