"""
Example script for generating a dataset using GenericPipelineDataset.
This example uses the local file output from the generic_pipeline_test_dataset example script to generate keywords for different tweets.
"""

import os
from datafast.schema.config import GenericPipelineDatasetConfig
from datafast.datasets import GenericPipelineDataset
from datafast.llms import OpenAIProvider, GeminiProvider, OllamaProvider, OpenRouterProvider
from datafast.logger_config import configure_logger
from dotenv import load_dotenv

# Load environment variables and configure logger
load_dotenv()
configure_logger()

PROMPT_TEMPLATE = """I will give you a tweet.
Generate a comma separated list of 3 keywords for the tweet. Avoid multi-word keywords.

Here is the tweet:
{tweet}

Your response should be in {language} and formatted in valid JSON with {num_samples} entry and all required fields."""

def main():
    # 1. Define the configuration
    config = GenericPipelineDatasetConfig(
        local_file_path="generic_pipeline_test_dataset.jsonl",
        input_columns=["tweet"],        # Input data for generation
        output_columns=["keywords"],    # Generated content columns
        num_samples_per_prompt=1,       # Generate 1 set per persona
        prompts=[PROMPT_TEMPLATE],
        output_file="keywords_extraction_z-ai_glm-4.6.jsonl",
        languages={"en": "English", "fr": "French"},
        sample_count=10
    )

    # 2. Initialize LLM providers
    providers = [
        # OllamaProvider(
        #     model_id="gemma3:27b-it-qat",
        #     api_base="https://xxxxxxx-11434.proxy.runpod.net", # if you have a runpod ollama instance
        #     temperature=1
        # )
        OpenRouterProvider(
            model_id="z-ai/glm-4.6",
        )
    ]

    # 3. Generate the dataset
    dataset = GenericPipelineDataset(config)
    num_expected_rows = dataset.get_num_expected_rows(providers)
    print(f"\nExpected number of rows: {num_expected_rows}")
    dataset.generate(providers)

    # 4. Print results summary
    print(f"\nGenerated {len(dataset.data_rows)} examples")
    print(f"Results saved to {config.output_file}")

    # 6. Optional: Push to HF hub
    USERNAME = "username"  # <--- Your hugging face username
    DATASET_NAME = "keywords_extraction_z-ai_glm-4.6"  # <--- Your hugging face dataset name
    url = dataset.push_to_hub(
        repo_id=f"{USERNAME}/{DATASET_NAME}",
        seed=20250816,
        shuffle=True,
    )
    print(f"\nDataset pushed to Hugging Face Hub: {url}")


if __name__ == "__main__":
    main()
