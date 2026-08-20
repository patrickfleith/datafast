"""OpenAI example showing unsupported parameter policies.

OpenAI's gpt-5 models are reasoning models served on the Responses API, which
rejects sampling controls such as ``temperature``. Datafast omits it and routes
the decision through the warn/quiet/fail policy.
"""

from __future__ import annotations

import warnings

from dotenv import load_dotenv

from datafast import openai


MODEL_ID = "gpt-5.4-mini"
PROMPT = "Explain OpenAI in one short sentence."


def run_case(policy: str) -> None:
    model = openai(
        MODEL_ID,
        temperature=0.5,
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
