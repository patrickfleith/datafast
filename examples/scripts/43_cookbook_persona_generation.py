"""Persona-generation cookbook: XSum article -> personas -> user prompts.

Demonstrates: Source.huggingface, Map, Filter, Sample, JSON-mode LLMSteps,
and prompt assets stored under docs/cookbook/assets.

Requires:
- MISTRAL_API_KEY
- PERSONA_COOKBOOK_HF_REPO_ID
- Hugging Face authentication via HF_TOKEN or a cached `huggingface_hub` login
- network access to Hugging Face and Mistral AI
"""

import os

from datafast import Filter, LLMStep, Map, Sample, Sink, Source, mistral

import litellm

litellm.suppress_debug_info = True


MODEL_ID = "mistral-small-2603"
SAMPLE_SIZE = 2
OUTPUT_PATH = "examples/outputs/43_persona_cookbook.jsonl"
HF_REPO_ID_ENV = "PERSONA_COOKBOOK_HF_REPO_ID"
HF_PRIVATE_ENV = "PERSONA_COOKBOOK_HF_PRIVATE"
TEXT_TO_PERSONA_PROMPT = "docs/cookbook/assets/text_to_persona.txt"
PERSONA_TO_PERSONA_PROMPT = "docs/cookbook/assets/persona_to_persona.txt"
PERSONA_TO_USER_PROMPT = "docs/cookbook/assets/persona_to_user_prompt.txt"


def add_word_count(record: dict) -> dict:
    return {**record, "word_count": len(record["document"].split())}


def keep_output_fields(record: dict) -> dict:
    return {
        "summary": record["summary"],
        "document": record["document"],
        "word_count": record["word_count"],
        "persona": record["persona"],
        "persona_basis": record["persona_basis"],
        "relationship_type": record["relationship_type"],
        "related_persona": record["related_persona"],
        "user_prompt": record["user_prompt"],
        "prompt_basis": record["prompt_basis"],
    }


def build_pipeline():
    model = mistral(MODEL_ID, temperature=0.7)

    return (
        Source.huggingface(
            "xsum",
            split="validation",
            columns=["document", "summary"],
        )
        >> Map(add_word_count).as_step("add_word_count")
        >> Filter(fn=lambda r: 300 <= r["word_count"] <= 500).as_step("filter_word_count")
        >> Sample(n=SAMPLE_SIZE, strategy="first").as_step("take_first_twenty")
        >> LLMStep(
            prompt=TEXT_TO_PERSONA_PROMPT,
            input_columns=["document"],
            output_columns=["persona", "persona_basis"],
            model=model,
            parse_mode="json",
            on_parse_error="raise",
        ).as_step("text_to_persona")
        >> LLMStep(
            prompt=PERSONA_TO_PERSONA_PROMPT,
            input_columns=["persona"],
            output_columns=["relationship_type", "related_persona"],
            model=model,
            parse_mode="json",
            on_parse_error="raise",
        ).as_step("persona_to_persona")
        >> LLMStep(
            prompt=PERSONA_TO_USER_PROMPT,
            input_columns=["related_persona"],
            output_columns=["user_prompt", "prompt_basis"],
            model=model,
            parse_mode="json",
            on_parse_error="raise",
        ).as_step("persona_to_user_prompt")
        >> Map(keep_output_fields).as_step("keep_output_fields")
        >> Sink.jsonl(OUTPUT_PATH)
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
