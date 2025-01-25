from datafast.datasets import TextClassificationDataset
from datafast.schema.config import ClassificationConfig, PromptExpansionConfig
from datafast.llms import create_provider, OpenAIProvider, AnthropicProvider, GoogleProvider
import os

def main():
    # 1. Configure the dataset generation
    config = ClassificationConfig(
        classes=[
            {
                "name": "concise",
                "description": "Concise text is characterized by clarity, precision, and brevity. \
                    It communicates ideas directly, in a very compact manner using only the words necessary to convey the intended message."
            },
            {
                "name": "verbose", 
                "description": "Verbose in text is often includes some redundancy, filler words, excessive qualifiers, \
                    unnecessary adjectives or adverbs, irrelevant information, and repetition of known context."
            }
        ],
        num_samples_per_prompt=10,
        output_file="concise_vs_verbose.jsonl",
        languages={
            'en': 'English',
            'fr': "French",
            'de': "German",
            'es': "Spanish"
        }
    )

    # 2. Create LLM providers (will use default models)
    providers = [
        OpenAIProvider(),  # default to gpt-4o-mini
        AnthropicProvider(),  # default to claude-3-5-haiku-latest
        GoogleProvider(),  # default to gemini-1.5-flash
    ]

    # 3. Generate the dataset
    dataset = TextClassificationDataset(config)
    dataset.generate(providers)

    # 4. Push to HF hub
    USERNAME = "patrickfleith"
    url = dataset.push_to_hub(
        repo_id=f"{USERNAME}/text-classification-datafast-demo",  # Replace with your username
        train_size=0.7,
        seed=20250125,
        shuffle=True,
    )
    print(f"\nDataset pushed to Hugging Face Hub: {url}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('secrets.env')
    main()
