"""Ollama example showing timeout and rpm_limit across multiple requests.

`timeout` matters most on the first call, where Ollama loads the model into
memory (a 12B model can take a while to warm up). `rpm_limit` is a client-side
throttle enforced by Datafast; a local Ollama server has no rate limit of its
own, but the control behaves the same as it does for hosted providers.
"""

import time

from dotenv import load_dotenv

from datafast import ollama


MODEL_ID = "gemma4:12b"
TIMEOUT_SECONDS = 120
RPM_LIMIT = 2
PROMPTS = [
    "Reply with exactly: request one acknowledged.",
    "Reply with exactly: request two acknowledged.",
    "Reply with exactly: request three acknowledged.",
]


def main() -> None:
    load_dotenv()

    model = ollama(
        MODEL_ID,
        temperature=0,
        timeout=TIMEOUT_SECONDS,
        rpm_limit=RPM_LIMIT,
    )

    print("Config")
    print("------")
    print(f"model: {MODEL_ID}")
    print(f"timeout: {TIMEOUT_SECONDS}s")
    print(f"rpm_limit: {RPM_LIMIT}")
    print()
    print(
        "This script sends three separate requests through one provider instance."
    )
    print(
        "With rpm_limit=2, the third request should pause for roughly one minute "
        "before Datafast sends it."
    )
    print()

    started = time.monotonic()
    for index, prompt in enumerate(PROMPTS, start=1):
        request_started = time.monotonic()
        response = model.generate(prompt=prompt)
        request_elapsed = time.monotonic() - request_started

        print(f"Request {index}")
        print(f"prompt: {prompt}")
        print(f"response: {response}")
        print(f"call_elapsed_seconds: {request_elapsed:.2f}")
        print()

    elapsed = time.monotonic() - started

    print("Notes")
    print("-----")
    print(f"total_elapsed_seconds: {elapsed:.2f}")
    print(
        "Datafast forwards timeout on each request and enforces rpm_limit on the "
        "provider instance before the next request is sent."
    )
    print(
        "Requests 1 and 2 should complete normally. Request 3 should be the one "
        "that clearly shows client-side throttling."
    )


if __name__ == "__main__":
    main()
