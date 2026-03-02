"""LLMStep loading a prompt template from a .txt file.

Demonstrates: prompt parameter pointing to a file path instead of inline string.
Requires: v2_examples/prompts/qa_prompt.txt (created by this project).
"""

from datafast_v2 import Source, LLMStep, Sink, openrouter
import litellm
litellm.suppress_debug_info = True


# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

pipeline = (
    Source.list([
        {"text": "The Great Wall of China was built over many centuries to protect against invasions."},
        {"text": "Photosynthesis converts sunlight into chemical energy in plants."},
    ])
    >> LLMStep(
        prompt="v2_examples/prompts/qa_prompt.txt",
        input_columns=["text"],
        output_columns=["question", "answer"],
        model=model,
        parse_mode="json",
    )
    >> Sink.jsonl("v2_examples/outputs/24_prompt_from_file.jsonl")
)

records = pipeline.run()
