from datafast.datasets import RawDataset
from datafast.schema.config import RawDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider, OllamaProvider


def main():
    # Configure the dataset generation with expansion config
    config = RawDatasetConfig(
        document_types=["personal blog"],
        topics=["artificial intelligence", "cybersecurity", "time series split", "RAG evaluations"],
        num_samples_per_prompt=3,
        output_file="tech_posts.jsonl",
        languages={"en": "English"},
        prompts=[
            "Generate {num_samples} {document_type} entries in {language_name} about {topic}. The emphasis should be a perspective from {{country}}. Answer with valid JSON."
        ],
        expansion=PromptExpansionConfig(
            placeholders={
                "country": ["United States", "Europe"]
            },
            combinatorial=True,
        ),
    )

    # 2. Create LLM providers with specific models
    providers = [
        OpenAIProvider(model_id="gpt-4.1-mini-2025-04-14"),
        # AnthropicProvider(model_id="claude-3-5-haiku-latest"),
        # GeminiProvider(model_id="gemini-2.0-flash"),
        # OllamaProvider(model_id="gemma3:4b"),
    ]

    # 3. Generate the dataset
    dataset = RawDataset(config)
    num_expected_rows = dataset.get_num_expected_rows(providers)
    print(f"\nExpected number of rows: {num_expected_rows}")
    dataset.generate(providers)

    # # 4. Push to HF hub (optional)
    # USERNAME = "YOUR_USERNAME"  # <--- Your hugging face username
    # DATASET_NAME = "YOUR_DATASET_NAME"  # <--- Your hugging face dataset name
    # url = dataset.push_to_hub(
    #     repo_id=f"{USERNAME}/{DATASET_NAME}",
    #     train_size=0.7,  # for a 80/20 train/test split, otherwise omit
    #     seed=20250304,
    #     shuffle=True,
    # )
    # print(f"\nDataset pushed to Hugging Face Hub: {url}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("secrets.env")
    main()