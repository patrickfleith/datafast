import itertools
import random
from math import prod

class ExpandPromptsError(Exception):
    """Base exception for expand_prompts function."""
    pass

def expand_prompts(
    prompt_templates: list[str],
    placeholders: dict[str, list[str]],
    combinatorial: bool = True,
    num_samples: int = 1
) -> list[tuple[str, dict[str, str]]]:
    """Expand template strings by filling in placeholders with provided values.

    This function generates variations of prompt templates by substituting placeholders
    with values using two possible strategies: combinatorial expansion or random sampling.

    Args:
        prompt_templates: List of template strings containing placeholders in {placeholder_name} format.
        placeholders: Dictionary mapping placeholder names to lists of possible values.
        combinatorial: If True, generates all possible combinations using cartesian product.
                      If False, generates random samples. Default is True.
        num_samples: Number of random variations to generate when combinatorial=False. Default is 1.

    Returns:
        list[tuple[str, dict[str, str]]]: List of tuples where each tuple contains:
            - expanded_prompt: Template with placeholders filled in
            - metadata_dict: Dictionary recording which values were chosen for each placeholder

    Examples:
        >>> templates = ["The {color} {animal} jumps"]
        >>> values = {
        ...     "color": ["red", "blue"],
        ...     "animal": ["fox", "rabbit"]
        ... }
        
        # Combinatorial expansion (all possible combinations):
        >>> expand_prompts(templates, values, combinatorial=True)
        [
            ("The red fox jumps", {"color": "red", "animal": "fox"}),
            ("The red rabbit jumps", {"color": "red", "animal": "rabbit"}),
            ("The blue fox jumps", {"color": "blue", "animal": "fox"}),
            ("The blue rabbit jumps", {"color": "blue", "animal": "rabbit"})
        ]

        # Random sampling:
        >>> expand_prompts(templates, values, combinatorial=False, num_samples=2)
        [
            ("The blue fox jumps", {"color": "blue", "animal": "fox"}),
            ("The red rabbit jumps", {"color": "red", "animal": "rabbit"})
        ]

    Notes:
        - Only placeholders that actually appear in a template are considered
        - Each template is processed independently
        - The function uses Python's string.format() for template expansion
        - When combinatorial=True, be cautious with large numbers of placeholders
          or values as the number of combinations grows exponentially
    """
    all_expanded = []

    for template in prompt_templates:
        # Identify placeholders actually used in this template
        used_keys = [k for k in placeholders if f"{{{k}}}" in template]

        if combinatorial:
            # cartesian product across used placeholders
            value_lists = [placeholders[k] for k in used_keys]
            for combo in itertools.product(*value_lists):
                combo_dict = dict(zip(used_keys, combo))
                filled_prompt = template.format(**combo_dict)
                all_expanded.append((filled_prompt, combo_dict))
        else:
            # random sampling approach
            for _ in range(num_samples):
                chosen = {}
                for k in used_keys:
                    chosen[k] = random.choice(placeholders[k])
                filled_prompt = template.format(**chosen)
                all_expanded.append((filled_prompt, chosen))

    return all_expanded
