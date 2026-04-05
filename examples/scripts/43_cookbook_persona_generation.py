"""Persona-generation cookbook: XSum article -> personas -> user prompts.

Demonstrates: Source.huggingface, Map, Filter, Sample, JSON-mode LLMSteps,
and prompt assets stored under docs/cookbook/assets.

Requires:
- OPENROUTER_API_KEY
- network access to Hugging Face and OpenRouter
"""

from datafast import Filter, LLMStep, Map, Sample, Sink, Source, openrouter

import litellm

litellm.suppress_debug_info = True


MODEL_ID = "nvidia/nemotron-3-super-120b-a12b"
OUTPUT_PATH = "examples/outputs/43_persona_cookbook.jsonl"
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


model = openrouter(MODEL_ID, temperature=0.7)

pipeline = (
    Source.huggingface(
        "xsum",
        split="validation",
        columns=["document", "summary"],
    )
    >> Map(add_word_count).as_step("add_word_count")
    >> Filter(fn=lambda r: 300 <= r["word_count"] <= 500).as_step("filter_word_count")
    >> Sample(n=5, strategy="first").as_step("take_first_five")
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

records = pipeline.run(batch_size=1)
