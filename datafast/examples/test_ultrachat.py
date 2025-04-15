from datafast.datasets import UltrachatDataset
from datafast.schema.config import UltrachatDatasetConfig
from datafast.llms import AnthropicProvider

def main():
    # 1. Define the configuration
    config = UltrachatDatasetConfig(
        domain="Materials Science for Wind Turbines",
        topics_and_subtopics={
            "Composite Materials": ["Fiber Reinforcement", "Resin Systems", "Fatigue Testing"],
            "Metal Alloys": ["Corrosion Resistance", "Structural Integrity", "Welding Properties"],
            "Polymers": ["UV Degradation", "Thermal Properties", "Impact Resistance"]
        },
        personas=[
            "Junior Engineer with 2 years of experience in the technical department",
            "Senior Materials Scientist specializing in durability under extreme conditions",
            "R&D Director evaluating materials for next-generation products"
        ],
        num_samples=3,
        max_turns=3,
        conversation_continuation_prob=0.25,
        output_file="wind_turbine_materials_conversations.jsonl",
        languages={"en": "English"}
    )

    # 2. Initialize LLM providers - using just one for simplicity
    providers = [
        AnthropicProvider(model_id="claude-3-5-haiku-latest"),
    ]

    # 3. Get expected number of rows
    dataset = UltrachatDataset(config)
    num_expected_rows = dataset.get_num_expected_rows(providers)
    print(f"\nExpected number of rows: {num_expected_rows}")
    
    # 4. Generate the dataset
    dataset.generate(providers)

    # 5. Print results summary
    print(f"\nGenerated {len(dataset.data_rows)} conversations")
    print(f"Results saved to {config.output_file}")

    # 6. Optional: Push to HF hub
    # USERNAME = "your_username"  # <--- Your hugging face username
    # DATASET_NAME = "wind_turbine_materials_dataset"  # <--- Your hugging face dataset name
    # url = dataset.push_to_hub(
    #     repo_id=f"{USERNAME}/{DATASET_NAME}",
    #     train_size=0.8,
    #     seed=42,
    #     shuffle=True,
    # )
    # print(f"\nDataset pushed to Hugging Face Hub: {url}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("secrets.env")
    main()