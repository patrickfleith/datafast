from datafast.datasets import TextDataset
from datafast.schema.config import TextDatasetConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GoogleProvider


def main():
    # 1. Configure the dataset generation
    config = TextDatasetConfig(
        document_types=["tech journalism blog", "personal blog", "MSc lecture notes"],
        topics=["technology", "artificial intelligence", "cloud computing"],
        num_samples_per_prompt=5,
        output_file="tech_posts.jsonl",
        languages={"en": "English", "fr": "French"},
    )

    # 2. Create LLM providers with specific models
    providers = [
        OpenAIProvider(model_id="gpt-4o-mini"),
        AnthropicProvider(model_id="claude-3-5-haiku-latest"),
        GoogleProvider(model_id="gemini-1.5-flash"),
    ]

    # 3. Generate the dataset
    dataset = TextDataset(config)
    dataset.generate(providers)

    # # 4. Push to HF hub
    # USERNAME = "patrickfleith"  # <--- Your hugging face username
    # DATASET_NAME = "YOUR_DATASET_NAME"  # <--- Your hugging face dataset name
    # url = dataset.push_to_hub(
    #     repo_id=f"{USERNAME}/{DATASET_NAME}",
    #     train_size=0.8,  # 80% for training
    #     seed=20250211,
    #     shuffle=True,
    # )
    # print(f"\nDataset pushed to Hugging Face Hub: {url}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("secrets.env")
    main()
