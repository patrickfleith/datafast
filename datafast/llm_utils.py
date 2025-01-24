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

def get_provider(provider: str, model_id: str, **kwargs):
    """Get an LLM provider instance.
    
    Args:
        provider (str): Provider name ('anthropic', 'google', or 'openai')
        model_id (str): Model identifier
        **kwargs: Additional provider-specific arguments
        
    Returns:
        LLMProvider: An initialized LLM provider instance
    """
    return create_provider(provider, model_id, **kwargs)
