from pydantic import BaseModel, Field, field_validator
from typing import Optional
import warnings


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
        default={"en": "English"},
        description="Language ISO codes and their corresponding names",
    )



class TextDatasetConfig(BaseModel):
    dataset_type: str = Field(default="text")
    
    # Text generation attributes
    document_types: list[str] = Field(
        default_factory=list,
        description="List of text generation document types. Required.",
    )

    topics: list[str] = Field(
        default_factory=list,
        description="List of text generation topics. Required.",
    )
       
    @field_validator('document_types')
    def validate_document_types(cls, v):
        if not v:
            raise ValueError("document_types is required and should be a list[str]")
        return v

    @field_validator('topics')
    def validate_topics(cls, v):
        if not v:
            raise ValueError("topics is required and should be a list[str]")
        return v
    
    @field_validator('num_samples_per_prompt')
    def validate_num_samples(cls, v):
        if v > 5:
            warnings.warn(
                "Values higher than 5 for num_samples_per_prompt are not recommended for raw text generation",
                UserWarning
            )
        return v

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
        default={"en": "English"},
        description="Language ISO codes and their corresponding names",
    )
