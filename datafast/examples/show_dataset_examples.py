"""
Demo: Show an example for each dataset type using the Gradio inspectors.

Run with:
    python show_dataset_examples.py

Requires: gradio
"""
from datafast.inspectors import (
    inspect_classification_dataset,
    inspect_mcq_dataset,
    inspect_preference_dataset,
    inspect_raw_dataset,
    inspect_ultrachat_dataset,
)
from datafast.schema.data_rows import (
    TextClassificationRow,
    MCQRow,
    PreferenceRow,
    TextRow,
    ChatRow,
)
from datafast.datasets import (
    ClassificationDataset,
    MCQDataset,
    PreferenceDataset,
    RawDataset,
    UltrachatDataset,
)
from datafast.schema.config import (
    ClassificationDatasetConfig,
    MCQDatasetConfig,
    PreferenceDatasetConfig,
)

# --- Classification Example ---
classification_row = TextClassificationRow(
    text="The trail is blocked by a fallen tree.",
    label="trail_obstruction",
    model_id="gpt-4.1-nano",
    metadata={"language": "en"},
)
classification_dataset = ClassificationDataset(
    ClassificationDatasetConfig(classes=[{"name": "trail_obstruction", "description": "Obstruction on the trail."}])
)
classification_row2 = TextClassificationRow(
    text="The trail is well maintained and easy to follow.",
    label="positive_conditions",
    model_id="claude-3-5-haiku-latest",
    metadata={"language": "en"},
)
classification_dataset.data_rows = [classification_row, classification_row2]

# --- MCQ Example ---
mcq_row = MCQRow(
    source_document="The Eiffel Tower is in Paris.",
    question="Where is the Eiffel Tower located?",
    correct_answer="Paris",
    incorrect_answer_1="London",
    incorrect_answer_2="Berlin",
    incorrect_answer_3="Rome",
    model_id="gemini-2.0-flash",
    metadata={"language": "en"},
)
mcq_config = MCQDatasetConfig(
    text_column="source_document",
    local_file_path="dummy.jsonl",  # Required by config, not used in this demo
)
mcq_dataset = MCQDataset(mcq_config)
mcq_row2 = MCQRow(
    source_document="The Amazon River is the second longest river in the world.",
    question="Which river is the second longest in the world?",
    correct_answer="Amazon River",
    incorrect_answer_1="Nile River",
    incorrect_answer_2="Yangtze River",
    incorrect_answer_3="Mississippi River",
    model_id="gpt-4.1-nano",
    metadata={"language": "en"},
)
mcq_dataset.data_rows = [mcq_row, mcq_row2]

# --- Preference Example ---
preference_row = PreferenceRow(
    input_document="Describe a recent Mars mission.",
    question="What was the main goal of the Mars 2020 mission?",
    chosen_response="To search for signs of ancient life and collect samples.",
    rejected_response="To launch a satellite.",
    chosen_model_id="claude-3-5-haiku-latest",
    rejected_model_id="gpt-4.1-nano",
    chosen_response_score=9,
    rejected_response_score=3,
    chosen_response_assessment="Accurate and detailed.",
    rejected_response_assessment="Too generic.",
    metadata={"language": "en"},
)
preference_dataset = PreferenceDataset(PreferenceDatasetConfig(input_documents=["Describe a recent Mars mission."]))
preference_row2 = PreferenceRow(
    input_document="Describe the Voyager 1 mission.",
    question="What is Voyager 1 known for?",
    chosen_response="It is the farthest human-made object from Earth, exploring interstellar space.",
    rejected_response="It is a Mars rover.",
    chosen_model_id="gemini-2.0-flash",
    rejected_model_id="gpt-4.1-nano",
    chosen_response_score=10,
    rejected_response_score=2,
    chosen_response_assessment="Factually correct and detailed.",
    rejected_response_assessment="Incorrect mission.",
    metadata={"language": "en"},
)
preference_dataset.data_rows = [preference_row, preference_row2]

# --- RawDataset Example ---
from datafast.schema.data_rows import TextRow
from datafast.schema.config import RawDatasetConfig, UltrachatDatasetConfig

raw_row1 = TextRow(
    text="SpaceX launched a new batch of Starlink satellites.",
    text_source="human",
    metadata={"date": "2025-06-30", "topic": "space"}
)
raw_row2 = TextRow(
    text="The James Webb Space Telescope captured new images of a distant galaxy.",
    text_source="synthetic",
    metadata={"date": "2025-06-29", "topic": "astronomy"}
)
raw_config = RawDatasetConfig(document_types=["news_article", "science_report"], topics=["space", "astronomy"])
raw_dataset = RawDataset(raw_config)
raw_dataset.data_rows = [raw_row1, raw_row2]

# --- UltrachatDataset Example ---
from datafast.schema.data_rows import ChatRow
ultrachat_row1 = ChatRow(
    opening_question="How can we reduce space debris?",
    persona="space policy expert",
    messages=[{"role": "user", "content": "What are current efforts to clean up space debris?"}, {"role": "assistant", "content": "There are several ongoing projects, such as RemoveDEBRIS and ClearSpace-1."}],
    model_id="gpt-4.1-nano",
    metadata={"language": "en"}
)
ultrachat_row2 = ChatRow(
    opening_question="What is the importance of the Moon missions?",
    persona="lunar geologist",
    messages=[{"role": "user", "content": "Why do we keep returning to the Moon?"}, {"role": "assistant", "content": "The Moon offers scientific insights and is a stepping stone for Mars exploration."}],
    model_id="gemini-2.0-flash",
    metadata={"language": "en"}
)
ultrachat_config = UltrachatDatasetConfig()
ultrachat_dataset = UltrachatDataset(ultrachat_config)
ultrachat_dataset.data_rows = [ultrachat_row1, ultrachat_row2]

if __name__ == "__main__":
    # print("Showing ClassificationDataset example...")
    # inspect_classification_dataset(classification_dataset)
    # print("Showing MCQDataset example...")
    # inspect_mcq_dataset(mcq_dataset)
    # print("Showing PreferenceDataset example...")
    # inspect_preference_dataset(preference_dataset)
    # print("Showing RawDataset example...")
    # inspect_raw_dataset(raw_dataset)
    # print("Showing UltrachatDataset example...")
    # inspect_ultrachat_dataset(ultrachat_dataset)
