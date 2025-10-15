"""
Example script for generating a dataset using GenericPipelineDataset.
This example uses the patrickfleith/FinePersonas-v0.1-100k-space-filtered dataset to generate tweets and CVs for different personas.
"""

import os
from datafast.schema.config import GenericPipelineDatasetConfig
from datafast.datasets import GenericPipelineDataset
from datafast.llms import OpenAIProvider, GeminiProvider, OllamaProvider


PROMPT_TEMPLATE = """I will give you a persona.
Generate {num_samples} texts in {language} with:
1. A tweet that this person might write (engaging, authentic to their character)
2. A short CV highlighting their background

Make sure the content reflects their personality and background authentically.
The CV should include higher education degree (and school/university they obtained it from), work experience (if any), and relevant skills, and a hobby.\

Here is the persona:
{persona}

Your response should be formatted in valid JSON with {num_samples} entries and all required fields."""

def main():
    # 1. Define the configuration
    config = GenericPipelineDatasetConfig(
        hf_dataset_name="patrickfleith/FinePersonas-v0.1-100k-space-filtered",
        input_columns=["persona"],                    # Input data for generation
        forward_columns=["summary_label"],            # Data to forward through
        output_columns=["tweet", "cv"],               # Generated content columns
        sample_count=5,                               # Process only 5 samples for testing
        num_samples_per_prompt=2,                     # Generate 1 set per persona
        prompts=[PROMPT_TEMPLATE],                    # Use the prompt template
        output_file="generic_pipeline_test_dataset.jsonl",
        languages={"en": "English", "fr": "French"}
    )

    # 2. Initialize LLM providers
    providers = [
        OpenAIProvider(
            model_id="gpt-5-mini-2025-08-07",
            temperature=1
            ),
        # AnthropicProvider(model_id="claude-haiku-4-5-20251001"),
        # GeminiProvider(model_id="gemini-2.5-flash-lite", rpm_limit=15),
        # OllamaProvider(model_id="gemma3:4b"),
    ]

    # 3. Generate the dataset
    dataset = GenericPipelineDataset(config)
    num_expected_rows = dataset.get_num_expected_rows(providers)
    print(f"\nExpected number of rows: {num_expected_rows}")
    dataset.generate(providers)

    # 4. Print results summary
    print(f"\nGenerated {len(dataset.data_rows)} examples")
    print(f"Results saved to {config.output_file}")

    # # 5. Show a sample of the generated data
    # if dataset.data_rows:
    #     print("\nSample generated row:")
    #     sample_row = dataset.data_rows[0]
    #     print(f"UUID: {sample_row.uuid}")
    #     print(f"Tweet: {getattr(sample_row, 'tweet', 'N/A')}")
    #     print(f"CV: {getattr(sample_row, 'cv', 'N/A')}")
    #     print(f"Persona: {getattr(sample_row, 'persona', 'N/A')}")
    #     print(f"Summary Label: {getattr(sample_row, 'summary_label', 'N/A')}")
    #     print(f"Model ID: {sample_row.model_id}")

    # 6. Optional: Push to HF hub
    USERNAME = "username"  # <--- Your hugging face username
    DATASET_NAME = "generic_pipeline_test_dataset_2"  # <--- Your hugging face dataset name
    url = dataset.push_to_hub(
        repo_id=f"{USERNAME}/{DATASET_NAME}",
        seed=20250816,
        shuffle=True,
    )
    print(f"\nDataset pushed to Hugging Face Hub: {url}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("secrets.env")
    main()
