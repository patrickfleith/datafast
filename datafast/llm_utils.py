from llms import google_generator, anthropic_generator, openai_generator

def get_messages(prompt: str) -> list[dict]:
    """Convert a single prompt into a message list format expected by LLM APIs.

    Args:
        prompt (str): The user's input prompt text

    Returns:
        list[dict]: A list of message dictionaries with system and user roles
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]


def get_generator_function(provider: str):
    """Get the appropriate generator function based on the provider."""
    generator_map = {
        'google': google_generator,
        'anthropic': anthropic_generator,
        'openai': openai_generator
    }
    return generator_map.get(provider)


