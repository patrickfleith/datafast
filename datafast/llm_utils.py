def get_messages(prompt: str) -> list[dict[str, str]]:
    """Convert a single prompt into a message list format expected by LLM APIs.

    Args:
        prompt (str): The user's input prompt text

    Returns:
        list[dict[str, str]]: A list of message dictionaries with system and user roles
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
