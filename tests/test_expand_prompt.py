from datafast.expanders import expand_prompts


def main():
    # Ensure reproducible random results (optional)

    prompt_templates = [
        "The {color} {animal} jumps.",
        "A {color} {animal} and a {second_animal} are friends.",
    ]

    placeholders = {
        "color": ["red", "blue", "green"],
        "animal": ["fox", "rabbit", "bear"],
        "second_animal": ["hedgehog", "cat", "dog"],
    }

    # 1) Default Combinatorial Test
    print("=== Default Combinatorial Expansion (max_samples=1000) ===")
    try:
        combinatorial_results = expand_prompts(
            prompt_templates=prompt_templates,
            placeholders=placeholders,
            combinatorial=True,
        )
        print(f"Total expansions (combinatorial): {len(combinatorial_results)}")
        for i, (prompt, meta) in enumerate(combinatorial_results, start=1):
            print(f"{i:2d}. Prompt: {prompt}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n")

    # 2) Combinatorial with small max_samples (should raise error)
    print("=== Combinatorial with small max_samples=10 ===")
    try:
        expand_prompts(
            prompt_templates=prompt_templates,
            placeholders=placeholders,
            combinatorial=True,
            max_samples=10,
        )
    except ValueError as e:
        print(f"Expected Error: {e}")

    print("\n")

    # 3) Random Sampling Test
    print("=== Random Sampling ===")
    num_random_samples = 6
    random_sampling_results = expand_prompts(
        prompt_templates=prompt_templates,
        placeholders=placeholders,
        combinatorial=False,
        num_random_samples=num_random_samples,
    )
    print(
        f"Total expansions (random, num_random_samples={num_random_samples}): \
            {len(random_sampling_results)}"
    )
    for i, (prompt, meta) in enumerate(random_sampling_results, start=1):
        print(f"{i:2d}. Prompt: {prompt}")

    print("\n")

    # 4) Random Sampling with custom max_samples
    print("=== Random Sampling with custom max_samples=4 ===")
    try:
        expand_prompts(
            prompt_templates=prompt_templates,
            placeholders=placeholders,
            combinatorial=False,
            num_random_samples=6,
            max_samples=4,
        )
    except ValueError as e:
        print(f"Expected Error: {e}")


if __name__ == "__main__":
    main()
