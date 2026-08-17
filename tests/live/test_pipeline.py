"""Live smoke test for the path users actually take.

Every other suite under `tests/live/` calls `ServedModel.generate()` directly,
which leaves the pipeline itself — prompt templating, the runner's batching, the
sink — covered only by mocks. This runs source → LLMStep → sink against each
provider that has a live suite, so a break between the step and the served model
shows up against a real endpoint.

One test per provider on purpose: the value here is the seam, not the breadth,
and the per-provider suites already cover provider behaviour in depth.
"""

import pytest

from datafast import LLMStep, ListSink, Source, anthropic, mistral, ollama, openai

# Mirrors the model ids in each provider's conftest. Kept small and cheap — the
# prompt asks for one word, so a weak model is enough and a mapping bug is still
# visible.
def _anthropic(require_api_key, require_ollama):
    require_api_key("ANTHROPIC_API_KEY")
    return anthropic(
        model_id="claude-haiku-4-5", temperature=0.0, max_completion_tokens=50
    )


def _openai(require_api_key, require_ollama):
    # No temperature: OPENAI_RESPONSES omits it, so setting one warns per call.
    require_api_key("OPENAI_API_KEY")
    return openai(model_id="gpt-5.4-mini", max_completion_tokens=50)


def _mistral(require_api_key, require_ollama):
    require_api_key("MISTRAL_API_KEY")
    return mistral(
        model_id="mistral-small-2603", temperature=0.0, max_completion_tokens=50
    )


def _ollama(require_api_key, require_ollama):
    require_ollama("qwen3:0.6b")
    return ollama(
        "qwen3:0.6b", temperature=0.0, max_completion_tokens=50, thinking=False
    )


PROVIDERS = [
    pytest.param(_anthropic, id="anthropic", marks=pytest.mark.anthropic),
    pytest.param(_openai, id="openai", marks=pytest.mark.openai),
    pytest.param(_mistral, id="mistral", marks=pytest.mark.mistral),
    pytest.param(_ollama, id="ollama", marks=pytest.mark.ollama),
]

# The assertion keys on a word the prompt dictates rather than on anything the
# model has to know, so a right answer in the wrong row cannot pass as a match.
WORDS = ["alpha", "bravo", "charlie"]

pytestmark = pytest.mark.live


@pytest.mark.parametrize("build_model", PROVIDERS)
def test_source_to_llm_step_to_sink(build_model, require_api_key, require_ollama):
    """Each source record yields one output row, in order, with its columns intact."""
    model = build_model(require_api_key, require_ollama)

    pipeline = (
        Source.list([{"word": word} for word in WORDS])
        >> LLMStep(
            prompt="Repeat this word exactly, in lowercase, and write nothing else: {word}",
            input_columns=["word"],
            output_column="echo",
            model=model,
        )
        >> ListSink()
    )

    output = pipeline.run()

    assert len(output) == len(WORDS)
    for word, record in zip(WORDS, output):
        assert record["word"] == word  # the input column survives the step
        assert word in record["echo"].lower()
        assert record["_model"] == model.model_id
