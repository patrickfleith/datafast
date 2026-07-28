"""OpenAI example with a batch of message lists."""

from dotenv import load_dotenv

from datafast import openai
from datafast.llm_utils import format_generated_responses


MODEL_ID = "gpt-5.4-mini"
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

    model = openai(MODEL_ID)
    responses = model.generate(messages=BATCH_MESSAGES)
    prompts = [messages[-1]["content"] for messages in BATCH_MESSAGES]
    print(format_generated_responses(prompts, responses))


if __name__ == "__main__":
    main()
