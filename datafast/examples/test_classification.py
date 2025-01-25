from datafast.datasets import TextClassificationDataset
from datafast.schema.config import ClassificationConfig, PromptExpansionConfig
from datafast.llms import create_provider, OpenAIProvider, AnthropicProvider
import os

def main():
    # 1. Configure the dataset generation
    config = ClassificationConfig(
        classes=[
            {
                "name": "machine_learning",
                "description": "Machine learning is a subset of artificial intelligence (AI) that focuses on developing algorithms and models that enable computers to learn patterns and make decisions or predictions based on data, without being explicitly programmed. It involves training models on datasets to identify relationships, recognize patterns, and improve performance over time, often categorized into supervised, unsupervised, and reinforcement learning."
            },
            {
                "name": "computer_science", 
                "description": "Computer science is the study of computers, computational systems, and algorithms, focusing on their theory, design, development, and application. It encompasses areas like programming, data structures, algorithms, artificial intelligence, databases, cybersecurity, computer architecture, software engineering, and more. Computer science aims to solve problems, automate tasks, and advance technology through computational thinking and innovation."
            }
        ],
        num_samples_per_prompt=4,  # Get 3 examples per prompt
        output_file="ml_vs_cs.jsonl",
    )

    # 2. Create LLM providers (will use default models)
    providers = [
        OpenAIProvider(),  # default to gpt-4o-mini
        AnthropicProvider(),  # default to claude-3-5-haiku-latest
    ]

    # 3. Generate the dataset
    dataset = TextClassificationDataset(config)
    dataset.generate(providers)

    print(f"Generated dataset saved to {config.output_file}")
    
    # Print first few examples
    for i, row in enumerate(dataset.data_rows[:5]):
        print(f"\nExample {i+1}:")
        print(f"Text: {row.text}")
        print(f"Label: {row.label}")
        print(f"Model: {row.model_id}")

    # 4. Push to Hugging Face Hub with train/test split
    try:
        # Using 80/20 split with stratification by label
        url = dataset.push_to_hub(
            repo_id="patrickfleith/ml-cs-classification",  # Replace with your username
            train_size=0.8,
            seed=42,  # For reproducibility
            shuffle=True,  # Optional: shuffle data for train_test_split
        )
        print(f"\nDataset pushed to Hugging Face Hub: {url}")
    except Exception as e:
        print(f"Error pushing to Hugging Face Hub: {e}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('secrets.env')
    main()
