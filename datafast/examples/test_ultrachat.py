from datafast.datasets import UltraChatDataset, ChatRow
from datafast.schema.config import UltraChatDatasetConfig
from datafast.llms import OpenAIProvider
from dotenv import load_dotenv

def main():
    config = UltraChatDatasetConfig(
        domain="Materials Science",
        topics_and_subtopics={
            "Polymers" : ["Design", "Testing"],
            "Alloys": ["Nickel", "Steel"]
        },
        personas=[
            "Master Student at Paris Saclay University, aspiring to pursue a PhD",
            "Head of Materials Science and Technologies Department at Airbus Defense and Space"
        ],
        num_samples=4,
        output_file="materials_science_example_instruction.jsonl",
    )

    # 2. Create LLM providers with specific models
    providers = [
        OpenAIProvider(model_id="gpt-4o-mini"),
    ]

    # 3. Generate the dataset
    dataset = UltraChatDataset(config)
    dataset.generate(providers)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("secrets.env")
    main()