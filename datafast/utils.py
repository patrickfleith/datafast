from datafast.schema.config import PromptExpansionConfig, ClassificationDatasetConfig, RawDatasetConfig, UltrachatDatasetConfig, MCQDatasetConfig, PreferenceDatasetConfig
from datafast.llms import LLMProvider

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