from datafast.datasets import RawDataset
from datafast.schema.config import RawDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider


def main():
    # 1. Configure the dataset generation
    config = RawDatasetConfig(
        document_types=[
            "space engineering textbook", 
            "spacecraft design justification document", 
            "personal blog of a space engineer"
        ],
        topics=[
            "Microgravity",
            "Vacuum",
            "Heavy Ions",
            "Thermal Extremes",
            "Atomic Oxygen",
            "Debris Impact",
            "Electrostatic Charging",
            "Propellant Boil-off",
            # ... You can pour hundreds of topics here. 8 is enough for this example
        ],
        num_samples_per_prompt=1,
        output_file="space_engineering_environment_effects_texts.jsonl",
        languages={"en": "English", "fr": "French"},
        prompts=[
            (
                "Generate {num_samples} section of a {document_type} in {language_name} " 
                "about the topic of {topic} in Space Engineering "
                "Target the content for {{expertise_level}} level readers."
            )
        ],
        expansion=PromptExpansionConfig(
            placeholders={
                "expertise_level": ["executives", "senior engineers", "PhD candidates"]
            },
            combinatorial=True,
        )
    )

    # 2. Create LLM providers with specific models
    providers = [
        OpenAIProvider(model_id="gpt-4.1-mini-2025-04-14"), # You may want to use stronger models
        AnthropicProvider(model_id="claude-3-5-haiku-latest"),
    ]

    # 3. Generate the dataset
    dataset = RawDataset(config)
    num_expected_rows = dataset.get_num_expected_rows(providers)
    print(f"Expected number of rows: {num_expected_rows}")
    dataset.generate(providers)

    # # 4. Push to HF hub (optional)
    # USERNAME = "patrickfleith"
    # DATASET_NAME = "space_engineering_environment_effects_texts"
    # url = dataset.push_to_hub(
    #     repo_id=f"{USERNAME}/{DATASET_NAME}",
    #     train_size=0.8,  # for a 80/20 train/test split, otherwise omit
    #     seed=20250319,
    #     shuffle=True,
    # )
    # print(f"\nDataset pushed to Hugging Face Hub: {url}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("secrets.env")
    main()