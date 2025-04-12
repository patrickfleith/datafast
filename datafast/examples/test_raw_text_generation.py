from datafast.datasets import TextDataset
from datafast.schema.config import TextDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider, OllamaProvider


def main():
    # Configure the dataset generation with expansion config
    config = TextDatasetConfig(
        document_types=["personal blog"],
        topics=["artificial intelligence", "cybersecurity", "time series split", "RAG evaluations"],
        num_samples_per_prompt=3,
        output_file="tech_posts.jsonl",
        languages={"en": "English"},
        prompts=[
            # "Generate {num_samples} {document_type} entries in {language_name} about {topic}. The emphasis should be a perspective from {{country}}. Answer with valid JSON."
            """I need {num_samples} text examples written in \
{language_name} that could have been written in a {document_type} related to {topic} with emphasis on {{country}}.
Make sure to properly format your response in valid JSON."""
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
        OpenAIProvider(model_id="gpt-4o-mini"),
        # AnthropicProvider(model_id="claude-3-5-haiku-latest"),
        # GeminiProvider(model_id="gemini-1.5-flash"),
        # OllamaProvider(model_id="gemma3:4b"),
    ]

    # 3. Generate the dataset
    dataset = TextDataset(config)
    dataset.generate(providers)

    # 4. Push to HF hub (optional)
    USERNAME = "patrickfleith"  # <--- Your hugging face username
    DATASET_NAME = "tech_posts"  # <--- Your hugging face dataset name
    url = dataset.push_to_hub(
        repo_id=f"{USERNAME}/{DATASET_NAME}",
        train_size=0.7,  # for a 80/20 train/test split, otherwise omit
        seed=20250304,
        shuffle=True,
    )
    print(f"\nDataset pushed to Hugging Face Hub: {url}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("secrets.env")
    main()