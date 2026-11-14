"""Ollama example showing unsupported parameter policies.

`ministral-3:3b` is not a thinking-capable model, so its profile does not map a
`reasoning_effort` control and the parameter flows through the warn/quiet/fail
policy. Thinking models (gemma4, deepseek-r1, qwen3, gpt-oss, ...) map it as a
first-class reasoning control instead — see the metadata example.
"""

from __future__ import annotations

import warnings

from dotenv import load_dotenv

from datafast import ollama


MODEL_ID = "ministral-3:3b"
PROMPT = "Explain what Ollama is in one short sentence."
REASONING_EFFORT = "high"


def run_case(policy: str) -> None:
    model = ollama(
        MODEL_ID,
        temperature=0,
        reasoning_effort=REASONING_EFFORT,
        unsupported_params=policy,
    )

    print(f"Policy: {policy}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            response = model.generate(prompt=PROMPT)
        except ValueError as exc:
            print("status: error")
            print(f"detail: {exc}")
        else:
            print("status: ok")
            print(f"text: {response.strip()}")

        print(f"warnings: {len(caught)}")
        for warning in caught:
            print(f"- {warning.message}")
    print()


def main() -> None:
    load_dotenv()

    for policy in ("warn", "quiet", "fail"):
        run_case(policy)


if __name__ == "__main__":
    main()
