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
