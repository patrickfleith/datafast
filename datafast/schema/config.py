from pydantic import BaseModel, Field
from typing import Optional

from pydantic import BaseModel, Field


class PromptExpansionConfig(BaseModel):
    placeholders: dict[str, list[str]] = {}
    combinatorial: bool = True
    num_samples: int = 1


class ClassificationConfig(BaseModel):
    """
    Configuration for generating a text classification dataset.
    """
    dataset_type: str = Field(default="text_classification")
    
    # The text classes
    classes: list[str] = Field(default_factory=list)

    # multi-label
    multilabel: bool = Field(default=False)

    # Prompt templates (strings) provided by the user; if empty, use defaults
    prompts: Optional[list[str]] = Field(default=None, description="Optional custom prompt templates")

    # Where to save the output
    output_file: str = Field(
        default="classification.jsonl",
        description="Path to save classification results"
    )

    # Expansion config
    expansion: PromptExpansionConfig = PromptExpansionConfig()

    languages: dict[str, str] = Field(default={"en": "English"}, description="Language ISO codes and their corresponding names")

    # Other relevant fields...