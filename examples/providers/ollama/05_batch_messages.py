"""Ollama example with a batch of message lists.

Same as the batch-prompts example, this uses Datafast's bounded-concurrency
fallback (Ollama has no native batch endpoint) and emits a UserWarning.
"""

from dotenv import load_dotenv

from datafast import ollama
from datafast.llm_utils import format_generated_responses


MODEL_ID = "gemma4:12b"
BATCH_MESSAGES = [
    [
        {
            "role": "system",
            "content": "You answer for engineers in one sentence.",
        },
        {
            "role": "user",
            "content": "What is prompt caching?",
        },
    ],
    [
        {
            "role": "user",
            "content": "What is structured output?",
        },
    ],
]


def main() -> None:
    load_dotenv()

    model = ollama(MODEL_ID, temperature=0, max_concurrent=1)
    responses = model.generate(messages=BATCH_MESSAGES)
    prompts = [messages[-1]["content"] for messages in BATCH_MESSAGES]
    print(format_generated_responses(prompts, responses))


if __name__ == "__main__":
    main()
