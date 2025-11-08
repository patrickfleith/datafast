from datafast.schema.config import PromptExpansionConfig, ClassificationDatasetConfig, RawDatasetConfig, UltrachatDatasetConfig, MCQDatasetConfig, PreferenceDatasetConfig, GenericPipelineDatasetConfig
from datafast.llms import LLMProvider
from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field, create_model
from loguru import logger
import time


def log_generation_progress(
    total_rows: int,
    provider_name: str,
    model_id: str,
    duration: float,
    item_type: str = "examples"
) -> None:
    """Log generation progress with provider and timing information.
    
    Args:
        total_rows: Total number of rows generated so far
        provider_name: Name of the LLM provider (e.g., "openai", "anthropic")
        model_id: Model identifier (e.g., "gpt-4", "claude-3-sonnet")
        duration: Duration in seconds for this generation step
        item_type: Type of items being generated (e.g., "examples", "chat conversations", "MCQs")
    
    Example:
        >>> log_generation_progress(25, "openai", "gpt-4", 3.2, "examples")
        # Logs: Generated and saved 25 examples total | Provider: openai | Model: gpt-4 | Duration: 3.2s
    """
    logger.success(
        f"Generated and saved {total_rows} {item_type} total | "
        f"Provider: {provider_name} | "
        f"Model: {model_id} | "
        f"Duration: {duration:.1f}s"
    )


def calculate_num_prompt_expansions(base_prompts: list[str], expansion_config: PromptExpansionConfig) -> int:
    """Calculate the number of prompt expansions based on the expansion configuration.
    Used to estimate the number of expected rows in the final dataset.
    
    Args:
        base_prompts: List of base prompt templates
        expansion_config: Configuration for prompt expansion
        
    Returns:
        int: Number of expanded prompts
    """
    placeholders = expansion_config.placeholders
    
    if expansion_config.combinatorial:
        # For combinatorial expansion, calculate all possible combinations
        num_expanded_prompts = 0
        
        for template in base_prompts:
            # Find which placeholder keys are used in this template
            used_keys = [k for k in placeholders if f"{{{k}}}" in template]
            if not used_keys:
                # Template with no placeholders counts as 1
                num_expanded_prompts += 1
                continue
                
            # Calculate combinations for this template
            template_combinations = 1
            for key in used_keys:
                values = placeholders.get(key, [])
                # If a key exists but has no values, default to 1
                template_combinations *= max(len(values), 1)
                
            num_expanded_prompts += template_combinations
    else:
        # For random sampling, use the configured number (capped by max_samples)
        num_expanded_prompts = min(
            expansion_config.num_random_samples,
            expansion_config.max_samples
        )
        
    return num_expanded_prompts


def _get_classficiation_specific_factors(config: ClassificationDatasetConfig) -> dict[str, int]:
    return {
        "num_classes": len(config.classes),
    }

def _get_classification_num_expected_rows(config: ClassificationDatasetConfig, llms: list[LLMProvider]) -> int:
    factors = _get_classficiation_specific_factors(config)
    num_llms = len(llms)
    if config.prompts is None:
        num_expanded_prompts = 1
    else:
        num_expanded_prompts = calculate_num_prompt_expansions(config.prompts, config.expansion)
    return (
        num_llms *
        len(config.languages or {"en": "English"}) *
        config.num_samples_per_prompt *
        factors["num_classes"] *
        num_expanded_prompts
    )


def _get_text_specific_factors(config: RawDatasetConfig) -> dict[str, int]:
    return {
        "num_document_types": len(config.document_types),
        "num_topics": len(config.topics),
    }


def _get_text_num_expected_rows(config: RawDatasetConfig, llms: list[LLMProvider]) -> int:
    factors = _get_text_specific_factors(config)
    num_llms = len(llms)
    if config.prompts is None:
        num_expanded_prompts = 1
    else:
        num_expanded_prompts = calculate_num_prompt_expansions(config.prompts, config.expansion)
    return (
        num_llms *
        len(config.languages or {"en": "English"}) *
        config.num_samples_per_prompt *
        factors["num_document_types"] *
        factors["num_topics"] *
        num_expanded_prompts
    )


def _get_ultrachat_specific_factors(config: UltrachatDatasetConfig) -> dict[str, int]:
    num_topic_subtopic_pairs = 0
    for _, value in config.topics_and_subtopics.items():
        num_topic_subtopic_pairs += len(value)
    return {
        "num_topic_subtopic_pairs": num_topic_subtopic_pairs,
    }


def _get_ultrachat_num_expected_rows(config: UltrachatDatasetConfig, llms: list[LLMProvider]) -> int:
    factors = _get_ultrachat_specific_factors(config)
    num_llms = len(llms)
    if config.question_generation_prompts is None:
        num_expanded_prompts = 1
    else:
        num_expanded_prompts = calculate_num_prompt_expansions(config.question_generation_prompts, config.expansion)
    return (
        num_llms *
        len(config.languages or {"en": "English"}) *
        config.num_samples *
        factors["num_topic_subtopic_pairs"] *
        num_expanded_prompts
    )


def _get_mcq_specific_factors(config: MCQDatasetConfig) -> dict[str, int]:
    return {"": None}  # There are no MCQ specific multipliers. Method here for consistency.


def _get_mcq_num_expected_rows(config: MCQDatasetConfig, llms: list[LLMProvider], source_data_num_rows: int) -> int:
    # factors = _get_mcq_specific_factors(config)  # Not specific factors
    if config.sample_count is not None:
        source_data_num_rows = min(source_data_num_rows, config.sample_count)
    num_llms = len(llms)
    if config.prompts is None:
        num_expanded_prompts = 1
    else:
        num_expanded_prompts = calculate_num_prompt_expansions(config.prompts, config.expansion)
    return (
        num_llms *
        len(config.languages or {"en": "English"}) *
        config.num_samples_per_prompt *
        source_data_num_rows *
        num_expanded_prompts
    )


def _get_preference_specific_factors(config: PreferenceDatasetConfig) -> dict[str, int]:
    return {"": None}  # There are no preference specific multipliers. Method here for consistency.

def _get_preference_num_expected_rows(config: PreferenceDatasetConfig, llms: list[LLMProvider]) -> int:
    # factors = _get_preference_specific_factors(config)  # Not specific factors
    num_llms = len(llms)
    num_docs = len(config.input_documents)
    num_questions = config.num_samples_per_prompt
    if config.question_generation_prompts is None:
        num_expanded_prompts = 1
    else:
        num_expanded_prompts = len(config.question_generation_prompts)
    return (
        num_llms *
        num_docs * 
        len(config.languages or {"en": "English"}) *
        num_questions *
        num_expanded_prompts
    )


def load_dataset_from_source(hf_dataset_name: str | None = None, 
                            local_file_path: str | None = None,
                            sample_count: int | None = None,
                            text_column: str = "text") -> list[dict]:
    """Shared utility to load dataset from Hugging Face or local file.
    
    Args:
        hf_dataset_name: Name of HuggingFace dataset
        local_file_path: Path to local file
        sample_count: Optional limit on number of samples
        text_column: Column name for text files (used by MCQDataset)
        
    Returns:
        List of dictionaries representing dataset rows
    """
    try:
        if hf_dataset_name:
            # Load from Hugging Face
            logger.info(f"Loading dataset from HuggingFace: {hf_dataset_name}")
            hf_dataset = load_dataset(hf_dataset_name)
            # Most datasets have a 'train' split, but fallback to first available split
            split_names = list(hf_dataset.keys())
            if not split_names:
                logger.error(f"No splits found in dataset {hf_dataset_name}")
                raise ValueError(f"No splits found in dataset {hf_dataset_name}")
                
            main_split = "train" if "train" in split_names else split_names[0]
            dataset = hf_dataset[main_split]
            # Convert to list of dicts for consistent interface
            dataset = [dict(row) for row in dataset]
            
        elif local_file_path:
            # Load from local file based on extension
            logger.info(f"Loading dataset from local file: {local_file_path}")
            file_ext = local_file_path.lower().split('.')[-1]
            
            if file_ext == 'csv':
                import pandas as pd
                df = pd.read_csv(local_file_path)
                dataset = df.to_dict('records')
                
            elif file_ext == 'txt':
                # For TXT files, use provided text_column name
                with open(local_file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                dataset = [{text_column: line} for line in lines]
                
            elif file_ext == 'parquet':
                import pandas as pd
                df = pd.read_parquet(local_file_path)
                dataset = df.to_dict('records')
                
            elif file_ext in ['jsonl', 'json']:
                import json
                dataset = []
                with open(local_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            dataset.append(json.loads(line))
                
            else:
                logger.error(f"Unsupported file extension: {file_ext}")
                raise ValueError(f"Unsupported file extension: {file_ext}. Supported extensions are: csv, txt, parquet, jsonl, json")
        else:
            logger.error("No dataset source specified")
            raise ValueError("Either hf_dataset_name or local_file_path must be specified")
            
        # Limit the number of samples if specified
        if sample_count is not None:
            dataset = dataset[:min(sample_count, len(dataset))]
            logger.info(f"Dataset loaded successfully | Rows: {len(dataset)}")
        else:
            logger.info(f"Dataset loaded successfully | Rows: {len(dataset)}")
            
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset | Error: {str(e)}")
        raise ValueError(f"Error loading dataset: {str(e)}")


def _get_generic_pipeline_specific_factors(config: GenericPipelineDatasetConfig) -> dict[str, int]:
    return {"": None}  # There are no generic pipeline specific multipliers. Method here for consistency.


def _get_generic_pipeline_num_expected_rows(config: GenericPipelineDatasetConfig, llms: list[LLMProvider]) -> int:
    """Calculate expected rows for GenericPipelineDataset including prompt expansions."""
    # Load source dataset to get row count
    logger.debug("Calculating expected rows for GenericPipelineDataset")
    source_dataset = load_dataset_from_source(
        hf_dataset_name=config.hf_dataset_name,
        local_file_path=config.local_file_path,
        sample_count=config.sample_count
    )
    source_data_num_rows = len(source_dataset)
    # Note: sample_count limit already applied in load_dataset_from_source()
    num_llms = len(llms)
    
    # Calculate prompt expansions
    if config.prompts is None:
        num_expanded_prompts = 1
    else:
        num_expanded_prompts = calculate_num_prompt_expansions(config.prompts, config.expansion)
    
    return (
        num_llms *
        len(config.languages or {"en": "English"}) *
        config.num_samples_per_prompt *
        source_data_num_rows *
        num_expanded_prompts
    )


def create_response_model(config: GenericPipelineDatasetConfig) -> type[BaseModel]:
    """
    Build a dynamic Pydantic model for GenericPipelineDataset response format.

    Args:
        config (GenericPipelineDatasetConfig): Config with output_columns specification.

    Returns:
        type[BaseModel]: Pydantic BaseModel class for structured LLM responses.
        If output_columns is None/empty, defaults to a single 'generated_text' field.
    """
    from typing import Any
    
    # Determine output fields
    if config.output_columns and len(config.output_columns) > 0:
        output_fields = config.output_columns
    else:
        output_fields = ["generated_text"]
    
    # Create field definitions for individual entry model
    entry_fields = {}
    for field_name in output_fields:
        entry_fields[field_name] = (str, Field(..., description=f"Generated content for {field_name}"))
    
    # Create the entry model
    EntryModel = create_model("GenericPipelineEntry", **entry_fields)
    
    # Create the response model with entries list
    ResponseModel = create_model(
        "GenericPipelineResponse",
        entries=(list[EntryModel], Field(..., description="List of generated entries"))
    )
    
    return ResponseModel


def create_generic_pipeline_row_model(config: GenericPipelineDatasetConfig) -> type[BaseModel]:
    """
    Build a dynamic Pydantic model for GenericPipelineRow based on configuration.

    Args:
        config (GenericPipelineDatasetConfig): Config with input_columns, forward_columns, and output_columns.

    Returns:
        type[BaseModel]: Dynamic Pydantic BaseModel class with all columns as separate fields.
    """
    from uuid import UUID, uuid4
    from datafast.schema.data_rows import GenericPipelineSource
    
    # Determine output fields
    if config.output_columns and len(config.output_columns) > 0:
        output_fields = config.output_columns
    else:
        output_fields = ["generated_text"]
    
    # Create field definitions for the row model in desired order
    row_fields = {
        # System fields first
        "uuid": (UUID, Field(default_factory=uuid4)),
    }
    
    # Add each output column as a separate field (right after uuid)
    for field_name in output_fields:
        row_fields[field_name] = (str, Field(..., description=f"Generated content for {field_name}"))
    
    # Processing metadata
    row_fields["model_id"] = (str | None, None)
    row_fields["pipeline_source"] = (GenericPipelineSource, GenericPipelineSource.SYNTHETIC)
    row_fields["language"] = (str | None, None)
    
    # Add each input column as a separate field
    for field_name in config.input_columns:
        row_fields[field_name] = (str, Field(..., description=f"Input data for {field_name}"))
    
    # Add each forward column as a separate field
    if config.forward_columns:
        for field_name in config.forward_columns:
            row_fields[field_name] = (str, Field(..., description=f"Forwarded data for {field_name}"))
    
    # Metadata last
    row_fields["metadata"] = (dict[str, str], Field(default_factory=dict))
    
    # Create the dynamic row model
    DynamicRowModel = create_model("DynamicGenericPipelineRow", **row_fields)
    
    return DynamicRowModel