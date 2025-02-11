from pydantic import BaseModel, Field
from typing import Optional


class PromptExpansionConfig(BaseModel):
    placeholders: dict[str, list[str]] = {}
    combinatorial: bool = True
    num_random_samples: int = 1
    max_samples: int = 1000


class ClassificationConfig(BaseModel):
    """
    Configuration for generating a text classification dataset.
    """

    dataset_type: str = Field(default="text_classification")

    # The text classes with their descriptions
    classes: list[dict[str, str | int]] = Field(
        default_factory=list,
        description="List of classification labels. Each label is a dict with \
            'label_id' (int), 'name' (str), and 'description' (str)",
    )

    # Prompt templates (strings) provided by the user; if empty, use defaults
    prompts: Optional[list[str]] = Field(
        default=None, description="Optional custom prompt templates"
    )

    num_samples_per_prompt: int = (
        5  # number of samples to generate simultaneously via LLM call.
    )

    # Where to save the output
    output_file: str = Field(
        default="classification.jsonl",
        description="Path to save classification results",
    )

    # Expansion config
    expansion: PromptExpansionConfig = PromptExpansionConfig()

    languages: dict[str, str] = Field(
        default_factory={"en": "English"},
        description="Language ISO codes and their corresponding names",
    )



class TextDatasetConfig(BaseModel):
    dataset_type: str = Field(default="text")

    # Text generation attributes
    text_attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Text generation attributes. Required: document_type, domain. \
            Optional: style, perspective, length, audience, format_structure, \
            additional_instructions"
    )
    
    # Prompt templates (strings) provided by the user; if empty, use defaults
    prompts: Optional[list[str]] = Field(
        default=None, description="Optional custom prompt templates"
    )

    num_samples_per_prompt: int = (
        5  # number of samples to generate simultaneously via LLM call.
    )

    # Where to save the output
    output_file: str = Field(
        default="text.jsonl",
        description="Path to save text results",
    )

    # Expansion config
    expansion: PromptExpansionConfig = PromptExpansionConfig()

    languages: dict[str, str] = Field(
        default_factory={"en": "English"},
        description="Language ISO codes and their corresponding names",
    )
