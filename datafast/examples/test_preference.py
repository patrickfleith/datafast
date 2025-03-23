"""
Example script for generating a Preference dataset with chosen and rejected responses.
"""

from datafast.schema.config import PreferenceDatasetConfig
from datafast.datasets import PreferenceDataset
from datafast.llms import OpenAIProvider, GoogleProvider

from datafast.examples.test_documents import TEST_DOCUMENTS


def main():
    # 1. Define the configuration
    config = PreferenceDatasetConfig(
        input_documents=TEST_DOCUMENTS,
        num_samples_per_prompt=2,  # Generate 2 questions per document
        languages={"en": "English", "fr": "French"},  # Generate in multiple languages
        llm_as_judge=True,  # Use LLM to judge and score responses
        output_file="preference_test_dataset.jsonl",
    )

    # 2. Initialize LLM providers
    question_gen_llm = GoogleProvider(model_id="gemini-1.5-flash")
    chosen_response_gen_llm = OpenAIProvider(model_id="gpt-4o-mini")
    rejected_response_gen_llm = GoogleProvider(model_id="gemini-1.5-flash")
    judge_llm = OpenAIProvider(model_id="gpt-4o-mini")

    # 3. Generate the dataset
    dataset = PreferenceDataset(config)
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
