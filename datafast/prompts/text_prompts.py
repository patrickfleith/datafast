from datafast.schema.config import TextDatasetConfig


DEFAULT_TEMPLATES = [
    """Your task is to generate {num_samples} texts written in \
{language_name} that could have been written in a {document_type} related to {domain}.
"""
]

def construct_text_generation_default_prompts(text_attributes: dict) -> list[str]:
    """
    Generate text-generation prompts using 2 mandatory attributes and
    several optional attributes. Only includes lines for optional attributes
    if they are provided.

    Args:
        text_attributes (dict): Dictionary containing text generation attributes
            Required: document_type, domain
            Optional: style, perspective, length, audience, format_structure,
                     additional_instructions

    Returns:
        list[str]: A list of final prompt strings ready for use with an LLM.

    Raises:
        ValueError: If required attributes (document_type, domain) are missing.
    """
    # Validate mandatory attributes
    required_attrs = ['document_type', 'domain']
    missing_attrs = [attr for attr in required_attrs if not text_attributes.get(attr)]
    if missing_attrs:
        raise ValueError(
            f"Missing required text_attributes in TextDatasetConfig: {', '.join(missing_attrs)}"
        )

    # Build optional lines that will be added to each template
    optional_lines = []
    if text_attributes.get('style'):
        optional_lines.append(f"Use a {text_attributes['style']} style.")
    if text_attributes.get('perspective'):
        optional_lines.append(f"Write from a {text_attributes['perspective']} perspective.")
    if text_attributes.get('length'):
        optional_lines.append(f"Length: {text_attributes['length']}.")
    if text_attributes.get('audience'):
        optional_lines.append(f"The intended audience of the text is {text_attributes['audience']}.")
    if text_attributes.get('format_structure'):
        optional_lines.append(f"Please follow this structure: {text_attributes['format_structure']}.")
    if text_attributes.get('additional_instructions'):
        optional_lines.append(f"Additionally, {text_attributes['additional_instructions']}.")

    # Combine each template with optional lines
    final_prompts = []
    for template in DEFAULT_TEMPLATES:
        lines = [template] + optional_lines
        final_prompts.append(" ".join(lines))

    return final_prompts