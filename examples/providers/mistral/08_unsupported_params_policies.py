"""Mistral example showing unsupported parameter policies.

Mistral chat models do not expose a reasoning-effort control, so
``reasoning_effort`` is unsupported and flows through the warn/quiet/fail policy.
"""

from __future__ import annotations

import warnings

from dotenv import load_dotenv

from datafast import mistral


MODEL_ID = "mistral-small-2603"
PROMPT = "Explain Mistral AI in one short sentence."
REASONING_EFFORT = "high"


def run_case(policy: str) -> None:
    model = mistral(
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
