"""
Example script for generating a Multiple Choice Question dataset.
"""

import os
from datafast.schema.config import MCQDatasetConfig
from datafast.datasets import MCQDataset
from datafast.llms import OpenAIProvider, AnthropicProvider, GoogleProvider, HuggingFaceProvider


def main():
    # 1. Define the configuration
    config = MCQDatasetConfig(
        hf_dataset_name="patrickfleith/space_engineering_environment_effects_texts",  # Stanford Question Answering Dataset
        text_column="text",    # Column containing the text to generate questions from
        sample_count=3,          # Process only 5 samples for testing
        num_samples_per_prompt=2,# Generate 2 questions per document
        min_document_length=100, # Skip documents shorter than 100 chars
        max_document_length=20000,# Skip documents longer than 20000 chars
        output_file="mcq_test_dataset.jsonl",
    )

    # 2. Initialize LLM providers
    providers = [
        # OpenAIProvider(model_id="gpt-4o-mini"),
        # AnthropicProvider(model_id="claude-3-5-haiku-latest"),
        GoogleProvider(model_id="gemini-2.0-flash"),
    ]

    # 3. Generate the dataset
    dataset = MCQDataset(config)
    dataset.generate(providers)

    # 4. Print results summary
    print(f"\nGenerated {len(dataset.data_rows)} MCQs")
    print(f"Results saved to {config.output_file}")

    # 5. Optional: Push to HF hub
    USERNAME = "patrickfleith"  # <--- Your hugging face username
    DATASET_NAME = "mcq_test_dataset"  # <--- Your hugging face dataset name
    url = dataset.push_to_hub(
        repo_id=f"{USERNAME}/{DATASET_NAME}",
        train_size=0.7,
        seed=20250125,
        shuffle=True,
    )
    print(f"\nDataset pushed to Hugging Face Hub: {url}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("secrets.env")
    main()
