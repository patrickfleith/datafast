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

    @field_validator("document_types")
    def validate_document_types(cls, v):
        if not v:
            raise ValueError("document_types is required and should be a list[str]")
        return v

    @field_validator("topics")
    def validate_topics(cls, v):
        if not v:
            raise ValueError("topics is required and should be a list[str]")
        return v

    @field_validator("num_samples_per_prompt")
    def validate_num_samples(cls, v):
        if v > 5:
            warnings.warn(
                "Values higher than 5 for num_samples_per_prompt are not recommended for raw text generation",
                UserWarning,
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


class UltraChatDatasetConfig(BaseModel):
    dataset_type: str = Field(default="instruction_dataset")

    conversation_continuation_prob: float = Field(
        default=0.5,
        description="Probability of continuing the conversation with a follow-up question",
        ge=0.0,
        le=1.0,
    )

    max_turns: int = Field(
        default=1,
        description="Maximum number of turns in generated Human-AI interaction (default to 1)",
        ge=1,
        le=10,
    )

    domain: str = Field(
        default="Science, Technology, Engineering, and Mathematics",
        description="Domain of the instruction dataset",
    )

    topics_and_subtopics: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Dictionary of topics and their corresponding subtopics",
    )

    personas: list[str] = Field(
        default_factory=list,
        description="List of personas",
    )

    num_samples: int = Field(
        default=10,
        description="Number of questions to generate for each topic and subtopic pair",
    )

    # Where to save the output
    output_file: str = Field(
        default="instruction_dataset.jsonl",
        description="Path to save instruction dataset results",
    )

    question_generation_prompts: Optional[list[str]] = Field(
        default=None,
        description="Optional custom prompt templates for question generation",
    )

    persona_question_reformulation_prompt: str = Field(
        default=None,
        description="Optional custom prompt template to reformulate \
                questions based on personas",
    )

    simulated_assistant_prompt: str = Field(
        default=None,
        description="Optional custom prompt template for the simulated \
                assistant",
    )

    user_system_prompt: str = Field(
        default=None,
        description="Optional custom system prompt for the AI to act \
                as a user",
    )

    user_followup_prompt: str = Field(
        default=None,
        description="Optional custom prompt template for the user's \
                follow-up message",
    )

    # Expansion config
    expansion: PromptExpansionConfig = PromptExpansionConfig()

    languages: dict[str, str] = Field(
        default={"en": "English"},
        description="Language ISO codes and their corresponding names",
    )
