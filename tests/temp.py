from datafast.datasets import TextClassificationDataset
from datafast.schema.config import ClassificationConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GoogleProvider
from datafast import utils


def test_get_num_expected_rows():
    """Test the get_num_expected_rows method with different configurations."""
    print("\n=== Testing get_num_expected_rows ===\n")
    
    # Test 1: Basic configuration with default prompts
    config1 = ClassificationConfig(
        classes=[
            {"name": "concise", "description": "Concise text description"},
            {"name": "verbose", "description": "Verbose text description"},
        ],
        num_samples_per_prompt=10,
        languages={"en": "English", "fr": "French"},
    )
    
    providers = [
        OpenAIProvider(model_id="gpt-4o-mini"),
        AnthropicProvider(model_id="claude-3-5-haiku-latest"),
        GoogleProvider(model_id="gemini-1.5-flash"),
    ]
    
    dataset1 = TextClassificationDataset(config1)
    expected_rows1 = dataset1.get_num_expected_rows(providers)
    print(f"Test 1 - Basic config with default prompts:")
    print(f"  - Classes: {len(config1.classes)}")
    print(f"  - Languages: {len(config1.languages)}")
    print(f"  - Providers: {len(providers)}")
    print(f"  - Samples per prompt: {config1.num_samples_per_prompt}")
    print(f"  - Expected rows: {expected_rows1}\n")
    
    # Test 2: With custom prompts
    config2 = ClassificationConfig(
        classes=[
            {"name": "concise", "description": "Concise text description"},
            {"name": "verbose", "description": "Verbose text description"},
        ],
        prompts=[
            "Generate {num_samples} examples of {label_name} ({label_description}) text in {language_name}",
            "Create {num_samples} examples of {label_name} ({label_description}) text in {language_name}",
        ],
        num_samples_per_prompt=5,
        languages={"en": "English"},
    )
    
    dataset2 = TextClassificationDataset(config2)
    expected_rows2 = dataset2.get_num_expected_rows(providers)
    print(f"Test 2 - With custom prompts:")
    print(f"  - Custom prompts: {len(config2.prompts)}")
    print(f"  - Expected rows: {expected_rows2}\n")
    
    # Test 3: With prompt expansions
    config3 = ClassificationConfig(
        classes=[
            {"name": "concise", "description": "Concise text description"},
            {"name": "verbose", "description": "Verbose text description"},
        ],
        prompts=[
            "Generate {num_samples} examples of {label_name} ({label_description}) text in {language_name} about {topic}."
        ],
        expansion=PromptExpansionConfig(
            placeholders={
                "topic": ["technology", "health", "education", "environment"]
            },
            combinatorial=False,
            num_random_samples=3,
            max_samples=100
        ),
        num_samples_per_prompt=5,
        languages={"en": "English"},
    )
    
    dataset3 = TextClassificationDataset(config3)
    expected_rows3 = dataset3.get_num_expected_rows(providers)
    print(f"Test 3 - With prompt expansions:")
    print(f"  - Base prompts: {len(config3.prompts)}")
    print(f"  - Placeholder values: {len(config3.expansion.placeholders['topic'])}")
    print(f"  - Expected rows: {expected_rows3}\n")
    
    # Test 4: Direct utility function test
    print(f"Test 4 - Direct utility function test:")
    direct_result = utils._get_classification_num_expected_rows(config3, providers)
    print(f"  - Direct utility result: {direct_result}")
    print(f"  - Dataset method result: {expected_rows3}")
    print(f"  - Match: {direct_result == expected_rows3}\n")


def main():
    # Run the test for get_num_expected_rows
    test_get_num_expected_rows()
    
    # Original dataset generation code
    config = ClassificationConfig(
        classes=[
            {
                "name": "concise",
                "description": "Concise text is characterized by clarity, precision, \
                 and brevity. It communicates ideas directly, in a very compact \
                manner using only the words necessary to convey the intended message.",
            },
            {
                "name": "verbose",
                "description": "Verbose in text is often includes some redundancy, \
                filler words, excessive qualifiers, unnecessary adjectives or \
                    adverbs, irrelevant information, and repetition of known context.",
            },
        ],
        num_samples_per_prompt=10,
        output_file="concise_vs_verbose.jsonl",
        languages={"en": "English", "fr": "French"},
    )
    
    # Create LLM providers (using the models specified in user rules)
    providers = [
        OpenAIProvider(model_id="gpt-4o-mini"),
        AnthropicProvider(model_id="claude-3-5-haiku-latest"),
        GoogleProvider(model_id="gemini-1.5-flash"),
    ]
    
    # Calculate expected rows before generation
    dataset = TextClassificationDataset(config)
    expected_rows = dataset.get_num_expected_rows(providers)
    print(f"\nMain dataset - Expected number of rows: {expected_rows}")
    
    # Uncomment to actually generate the dataset
    # print("\nGenerating dataset...")
    # dataset.generate(providers)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("secrets.env")
    main()