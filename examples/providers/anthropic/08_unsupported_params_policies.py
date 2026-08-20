"""Anthropic example showing unsupported parameter policies.

``previous_response_id`` is a Responses-API concept. Anthropic runs on the chat
endpoint, so it is unsupported and flows through the warn/quiet/fail policy.
"""

from __future__ import annotations

import warnings

from dotenv import load_dotenv

from datafast import anthropic


MODEL_ID = "claude-haiku-4-5"
PROMPT = "Explain Anthropic in one short sentence."


def run_case(policy: str) -> None:
    model = anthropic(
        MODEL_ID,
        temperature=0,
        unsupported_params=policy,
    )

    print(f"Policy: {policy}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            response = model.generate(prompt=PROMPT, previous_response_id="resp_demo")
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
