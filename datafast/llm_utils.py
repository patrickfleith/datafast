from __future__ import annotations

from collections.abc import Sequence


def get_messages(prompt: str, system_message: str = "You are a helpful assistant.") -> list[dict[str, str]]:
    """Convert a single prompt into a message list format expected by LLM APIs.

    Args:
        prompt (str): The user's input prompt text.
        system_message (str, optional): The system message to include. Defaults to "You are a helpful assistant."

    Returns:
        list[dict[str, str]]: A list of message dictionaries with system and user roles
    """
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]


def format_generated_responses(
    prompts: str | Sequence[str],
    responses: str | Sequence[str],
) -> str:
    """Return a readable string for one or many prompt/response pairs."""
    prompt_items = [prompts] if isinstance(prompts, str) else list(prompts)
    response_items = [responses] if isinstance(responses, str) else list(responses)

    if len(prompt_items) != len(response_items):
        raise ValueError("prompts and responses must have the same length")

    sections = [
        _format_response_section(prompt, response, index, total=len(prompt_items))
        for index, (prompt, response) in enumerate(
            zip(prompt_items, response_items, strict=True),
            start=1,
        )
    ]
    return "\n\n".join(sections)


def _format_response_section(
    prompt: str,
    response: str,
    index: int,
    *,
    total: int,
) -> str:
    lines = []
    if total > 1:
        lines.append(f"Example {index}")
    lines.extend(
        [
            "Prompt",
            "------",
            prompt,
            "",
            "Response",
            "--------",
            response,
        ]
    )
    return "\n".join(lines)
