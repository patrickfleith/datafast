import itertools
import random
from math import prod


def expand_prompts(
    prompt_templates: list[str],
    placeholders: dict[str, list[str]],
    combinatorial: bool = True,
    num_random_samples: int = 1,
    max_samples: int = 1000,
) -> list[tuple[str, dict[str, str]]]:
    """Generate expanded prompts by filling template placeholders with provided values.

    This versatile function supports two modes of operation:
    1. Combinatorial Mode (default): Generates all possible combinations per template
    2. Random Sampling Mode: Creates a specified number of random expansions a
        cross templates

    Key Features:
    - Handles multiple templates simultaneously
    - Supports both exhaustive and randomized sampling
    - Provides detailed metadata about chosen values
    - Built-in safety limits to prevent memory issues
    - Preserves templates without placeholders

    Args:
        prompt_templates: Templates containing {placeholder} format strings
            Example: ["The {color} {animal} jumps", "I love {food}"]
        placeholders: Mapping of placeholder names to their possible values
            Example: {"color": ["red", "blue"], "animal": ["fox", "dog"], "food":
                ["pizza"]}
        combinatorial: If True, generates all possible combinations (default)
            If False, performs random sampling based on num_random_samples
        num_random_samples: Number of random samples to generate when
            combinatorial=False
            Ignored when combinatorial=True
        max_samples: Safety limit for total number of generated prompts
            Applies to both combinatorial and random sampling modes

    Returns:
        A list of tuples, each containing:
        - expanded_prompt: The fully expanded template string
        - metadata: Dictionary mapping used placeholders to their chosen values

        Example:
        [
            ("The red fox jumps", {"color": "red", "animal": "fox"}),
            ("The blue dog jumps", {"color": "blue", "animal": "dog"}),
            ("I love pizza", {"food": "pizza"})
        ]

    Raises:
        ValueError: If the total possible combinations or requested samples exceed \
            max_samples

    Performance Notes:
    - Memory usage scales with the number of combinations/samples
    - For large combinatorial spaces, consider using random sampling instead
    """
    if not combinatorial and num_random_samples > max_samples:
        raise ValueError(
            f"num_random_samples ({num_random_samples}) cannot exceed max_samples\
                 ({max_samples})"
        )

    if combinatorial:
        total_combinations = sum(
            prod(len(placeholders[k]) for k in placeholders if f"{{{k}}}" in template)
            or 1
            for template in prompt_templates
        )

        if total_combinations > max_samples:
            raise ValueError(
                f"Total possible combinations ({total_combinations}) exceeds "
                f"max_samples ({max_samples}). Consider using combinatorial=False "
                f"with num_random_samples to randomly sample instead."
            )

    expanded_prompts = []

    if combinatorial:
        for template in prompt_templates:
            used_keys = [k for k in placeholders if f"{{{k}}}" in template]

            if not used_keys:
                expanded_prompts.append((template, {}))
                continue

            value_lists = [placeholders[k] for k in used_keys]
            for values in itertools.product(*value_lists):
                combo_dict = dict(zip(used_keys, values))
                expanded_prompts.append((template.format(**combo_dict), combo_dict))
    else:
        num_templates = len(prompt_templates)
        for i in range(num_random_samples):
            template = prompt_templates[i % num_templates]
            used_keys = [k for k in placeholders if f"{{{k}}}" in template]
            chosen_values = {key: random.choice(placeholders[key]) for key in used_keys}
            expanded_prompts.append((template.format(**chosen_values), chosen_values))

    return expanded_prompts
