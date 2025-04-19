"""
Example script for generating a Preference dataset with chosen and rejected responses.
"""

import json
from pathlib import Path

from datafast.schema.config import PreferenceDatasetConfig
from datafast.datasets import PreferenceDataset 
from datafast.llms import OpenAIProvider, GeminiProvider, AnthropicProvider


def load_documents_from_jsonl(jsonl_path: str | Path) -> list[str]:
    """
    Load documents from a JSONL file where the document text is stored in the 'text' key.
    
    Args:
        jsonl_path: Path to the JSONL file
    
    Returns:
        List of document strings
    """
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'text' in data:
                documents.append(data['text'])
    return documents


def main():
    # Load documents from JSONL file
    jsonl_path = Path(__file__).parent / 'data/preferences/nasa_lsi_sample.jsonl'
    documents = load_documents_from_jsonl(jsonl_path)
    
    # 1. Define the configuration
    config = PreferenceDatasetConfig(
        input_documents=documents,
        num_samples_per_prompt=3,  # Generate 3 questions per document
        languages={"en": "English"},  # Generate in English
        llm_as_judge=True,  # Use LLM to judge and score responses
        output_file="nasa_lessons_learned_dataset.jsonl",
    )

    # 2. Initialize LLM providers
    question_gen_llm = OpenAIProvider(model_id="gpt-4.1-mini")
    chosen_response_gen_llm = AnthropicProvider(model_id="claude-3-7-sonnet-latest")
    rejected_response_gen_llm = GeminiProvider(model_id="gemini-2.0-flash")
    judge_llm = OpenAIProvider(model_id="gpt-4.1")

    # 3. Generate the dataset
    dataset = PreferenceDataset(config)
    num_expected_rows = dataset.get_num_expected_rows(llms=[question_gen_llm])
    print(f"\nExpected number of rows: {num_expected_rows}")
    dataset.generate(
        question_gen_llm=question_gen_llm,
        chosen_response_gen_llm=chosen_response_gen_llm,
        rejected_response_gen_llm=rejected_response_gen_llm,
        judge_llm=judge_llm
    )

    # 4. Print results summary
    print(f"\nGenerated {len(dataset.data_rows)} preference pairs")
    print(f"Results saved to {config.output_file}")

    # 5. Display a sample of the generated data
    if dataset.data_rows:
        sample = dataset.data_rows[0]
        print("\nSample preference pair:")
        print(f"Question: {sample.question}")
        print(f"Chosen model: {sample.chosen_model_id}")
        print(f"Rejected model: {sample.rejected_model_id}")
        if sample.chosen_response_score is not None:
            print(f"Chosen response score: {sample.chosen_response_score}")
            print(f"Rejected response score: {sample.rejected_response_score}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("secrets.env")
    main()
