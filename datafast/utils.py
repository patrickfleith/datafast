from datafast.schema.config import PromptExpansionConfig, ClassificationConfig
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


def _get_classficiation_specific_factors(config: ClassificationConfig) -> dict[str, int]:
    return {
        "num_classes": len(config.classes),
    }

def _get_classification_num_expected_rows(config: ClassificationConfig, llms: list[LLMProvider]) -> int:
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
    