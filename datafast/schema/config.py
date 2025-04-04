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
    classes: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of classification labels. Each label is a dict with \
            'name' (str), and 'description' (str)",
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
    
    @field_validator("prompts")
    def validate_prompts(cls, v):
        if v is not None:
            required_placeholders = ["{num_samples}", "{language_name}", "{label_name}", "{label_description}"]
            for i, prompt in enumerate(v):
                missing_placeholders = [p for p in required_placeholders if p not in prompt]
                if missing_placeholders:
                    raise ValueError(
                        f"Prompt at index {i} is missing required placeholders: {', '.join(missing_placeholders)}. "
                        f"All prompts must contain: {', '.join(required_placeholders)}"
                    )
        return v


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
    
    @field_validator("prompts")
    def validate_prompts(cls, v):
        if v is not None:
            required_placeholders = ["{num_samples}", "{language_name}", "{document_type}", "{topic}"]
            for i, prompt in enumerate(v):
                missing_placeholders = [p for p in required_placeholders if p not in prompt]
                if missing_placeholders:
                    raise ValueError(
                        f"Prompt at index {i} is missing required placeholders: {', '.join(missing_placeholders)}. "
                        f"All prompts must contain: {', '.join(required_placeholders)}"
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

    persona_question_reformulation_prompt: Optional[str] = Field(
        default=None,
        description="Optional custom prompt template to reformulate \
                questions based on personas",
    )

    simulated_assistant_prompt: Optional[str] = Field(
        default=None,
        description="Optional custom prompt template for the simulated \
                assistant",
    )

    user_system_prompt: Optional[str] = Field(
        default=None,
        description="Optional custom system prompt for the AI to act \
                as a user",
    )

    user_followup_prompt: Optional[str] = Field(
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

    @field_validator("question_generation_prompts")
    def validate_prompts(cls, v):
        if v is not None:
            required_placeholders = ["{num_samples}", "{language_name}", "{domain}", "{topic}", "{subtopic}"]
            for i, prompt in enumerate(v):
                missing_placeholders = [p for p in required_placeholders if p not in prompt]
                if missing_placeholders:
                    raise ValueError(
                        f"Prompt at index {i} is missing required placeholders: {', '.join(missing_placeholders)}. "
                        f"All prompts must contain: {', '.join(required_placeholders)}"
                    )
        return v


class MCQDatasetConfig(BaseModel):
    """
    Configuration for generating multiple choice questions from text in a Hugging Face dataset.
    Each question has one correct answer and three plausible but incorrect answers.
    """
    dataset_type: str = Field(default="mcq_dataset")
    
    # Hugging Face dataset information
    hf_dataset_name: str = Field(
        ...,  # required field
        description="Name of the Hugging Face dataset to use"
    )
    
    text_column: str = Field(
        ...,  # required field
        description="Column name containing the text to generate questions from"
    )
    
    # MCQ Generation parameters
    num_samples_per_prompt: int = Field(
        default=3,
        description="Number of questions to generate for each text"
    )
    
    sample_count: Optional[int] = Field(
        default=None,
        description="Optional number of samples to process from the dataset"
    )

    min_document_length: int = Field(
        default=100,
        description="Minimum number of characters below which documents will be skipped"
    )

    max_document_length: int = Field(
        default=10000,
        description="Maximum number of characters above which documents will be skipped"
    )
    
    # Where to save the output
    output_file: str = Field(
        default="mcq_dataset.jsonl",
        description="Path to save MCQ dataset results"
    )
    
    # Optional custom prompts
    prompts: Optional[list[str]] = Field(
        default=None, 
        description="Optional custom prompt templates"
    )

    distractor_prompt: Optional[str] = Field(
        default=None,
        description="Optional custom distractor prompt template"
    )
    
    # Standard config options
    expansion: PromptExpansionConfig = PromptExpansionConfig()
    
    languages: dict[str, str] = Field(
        default={"en": "English"},
        description="Language ISO codes and their corresponding names"
    )
    
    @field_validator("hf_dataset_name")
    def validate_dataset_name(cls, v):
        if not v:
            raise ValueError("hf_dataset_name is required")
        return v
    
    @field_validator("text_column")
    def validate_text_column(cls, v):
        if not v:
            raise ValueError("text_column is required")
        return v


class PreferenceDatasetConfig(BaseModel):
    dataset_type: str = Field(default="preference_dataset")

    # Input documents
    input_documents: list[str] = Field(
        default_factory=list,
        description="List of input documents from which questions will be generated"
    )
    
    num_samples_per_prompt: int = Field(
        default=3,
        description="Number of questions generated per persona/document pair"
    )

    question_generation_prompts: Optional[list[str]] = Field(
        default=None,
        description="Optional custom prompt templates for question generation",
    )

    chosen_response_generation_prompt: Optional[str] = Field(
        default=None,
        description="Optional custom prompt template for generation of the chosen response",
    )

    rejected_response_generation_prompt: Optional[str] = Field(
        default=None,
        description="Optional custom prompt template for generation of the rejected response",
    )

    output_file: str = Field(
        default="preference_dataset.jsonl",
        description="Path to save preference dataset results"
    )

    # Expansion config - Not yet supported for PreferenceDataset
    expansion: PromptExpansionConfig = PromptExpansionConfig()
    
    @field_validator('expansion')
    def expansion_not_supported(cls, v, info):
        if v and (v.placeholders or v.combinatorial or v.num_random_samples != 0):
            raise ValueError("Expansion is not yet supported for PreferenceDataset")
        return v

    languages: dict[str, str] = Field(
        default={"en": "English"},
        description="Language ISO codes and their corresponding names",
    )

    evol_instruct: bool = Field(
        default=False,
        description="Whether to use evolutionary instruction refinement"
    )
    
    llm_as_judge: bool = Field(
        default=False,
        description="Whether to use an LLM as judge for preference pairs scoring"
    )
    
    # Conditional fields for evol_instruct
    evolution_prompt: Optional[str] = Field(
        default=None,
        description="Prompt template for evolutionary instruction refinement (required when evol_instruct=True)"
    )
    
    # Conditional fields for llm_as_judge
    judge_prompt: Optional[str] = Field(
        default=None,
        description="Prompt template for the LLM judge (required when llm_as_judge=True)"
    )
    
    @field_validator("evolution_prompt")
    def validate_evolution_prompt(cls, v, info):
        values = info.data
        if values.get("evol_instruct", False) and not v:
            raise ValueError("evolution_prompt is required when evol_instruct is True")
        return v
    
    @field_validator("judge_prompt")
    def validate_judge_prompt(cls, v, info):
        values = info.data
        if values.get("llm_as_judge", False) and not v:
            raise ValueError("judge_prompt is required when llm_as_judge is True")
        return v
