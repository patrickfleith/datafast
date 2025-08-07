"""
Example script for generating MCQ questions from the AR6 dataset using context information.
This script demonstrates the use of the context_column parameter to enhance question generation.
"""

import os
import json
import random
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv("secrets.env")

from datafast.schema.config import MCQDatasetConfig
from datafast.datasets import MCQDataset
from datafast.llms import OpenAIProvider

def main():
    # 1. Create a temporary filtered version of the dataset
    ar6_file_path = Path("datafast/examples/data/mcq/ar6.jsonl")
    filtered_file_path = Path("datafast/examples/data/mcq/ar6_filtered.jsonl")
    
    # Read the ar6.jsonl file
    with open(ar6_file_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    # Filter for rows where chunk_grade is "OK" or "GREAT"
    filtered_data = [row for row in data if row.get("chunk_grade") in ["OK", "GREAT"]]
    
    # Randomly select 10 examples
    selected_data = random.sample(filtered_data, min(10, len(filtered_data)))
    
    # Write the selected data to a temporary file
    with open(filtered_file_path, "w") as f:
        for row in selected_data:
            f.write(json.dumps(row) + "\n")
    
    print(f"Selected {len(selected_data)} examples from AR6 dataset")
    
    # 2. Create MCQ dataset config
    config = MCQDatasetConfig(
        local_file_path=str(filtered_file_path),
        text_column="chunk_text",           # Column containing the text to generate questions from
        context_column="document_summary",  # Column containing context information
        num_samples_per_prompt=2,           # Generate 2 questions per document
        min_document_length=100,            # Skip documents shorter than 100 chars
        max_document_length=20000,          # Skip documents longer than 20000 chars
        sample_count=len(selected_data),    # Number of samples to process
        output_file="mcq_ar6_contextual_dataset.jsonl",
    )

    # 3. Initialize OpenAI provider with gpt-4.1-mini
    providers = [
        OpenAIProvider(model_id="gpt-4.1-mini"),
    ]

    # 4. Generate the dataset
    dataset = MCQDataset(config)
    num_expected_rows = dataset.get_num_expected_rows(providers, source_data_num_rows=len(selected_data))
    print(f"\nExpected number of rows: {num_expected_rows}")
    dataset.generate(providers)

    # 5. Print results summary
    print(f"\nGenerated {len(dataset.data_rows)} MCQs")
    print(f"Results saved to {config.output_file}")
    
    # 6. Cleanup temporary file
    os.remove(filtered_file_path)
    print(f"Cleaned up temporary file {filtered_file_path}")
    # # 5. Optional: Push to HF hub
    # USERNAME = "your_username"  # <--- Your hugging face username
    # DATASET_NAME = "your_dataset_name"  # <--- Your hugging face dataset name
    # url = dataset.push_to_hub(
    #     repo_id=f"{USERNAME}/{DATASET_NAME}",
    #     train_size=0.7,
    #     shuffle=True,
    #     upload_card=True,
    # )
    # print(f"\nDataset pushed to Hugging Face Hub: {url}")

    dataset.inspect()

if __name__ == "__main__":
    main()
