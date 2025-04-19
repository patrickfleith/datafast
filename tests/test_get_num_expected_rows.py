import pytest
from unittest.mock import Mock
from datafast.datasets import ClassificationDataset
from datafast.schema.config import ClassificationDatasetConfig, PromptExpansionConfig
from datafast import utils

from dotenv import load_dotenv

load_dotenv('secrets.env')

@pytest.fixture
def providers():
    # Mock providers as we only need to count them
    return [Mock(), Mock(), Mock()]

def test_basic_expected_rows_formula(providers):
    # Test the expected rows calculation with default prompts, multiple classes, and languages.
    config = ClassificationDatasetConfig(
        classes=[
            {"name": "concise", "description": "Concise text description"},
            {"name": "verbose", "description": "Verbose text description"},
        ],
        num_samples_per_prompt=10,
        languages={"en": "English", "fr": "French"},
    )
    dataset = ClassificationDataset(config)
    expected = (
        len(providers)
        * 1  # Use 1 for the single default prompt when config.prompts is None
        * len(config.classes)
        * len(config.languages)
        * config.num_samples_per_prompt
    )
    assert dataset.get_num_expected_rows(providers) == expected

def test_custom_prompts_expected_rows_numeric(providers):
    # Test the expected rows calculation when custom prompts are provided.
    config = ClassificationDatasetConfig(
        classes=[
            {"name": "concise", "description": "Concise text description"},
            {"name": "verbose", "description": "Verbose text description"},
        ],
        prompts=[
            "Generate {num_samples} examples of {label_name} ({label_description}) text in {language_name}",
            "Create {num_samples} examples of {label_name} ({label_description}) text in {language_name}",
        ],
        num_samples_per_prompt=5,
        languages={"en": "English"},
    )
    dataset = ClassificationDataset(config)
    # 2 classes * 2 prompts * 1 language * 5 samples * 3 providers = 60
    assert dataset.get_num_expected_rows(providers) == 60

def test_prompt_expansions_expected_rows_numeric(providers):
    # Test the expected rows calculation when prompt expansion (random sampling) is used.
    config = ClassificationDatasetConfig(
        classes=[
            {"name": "concise", "description": "Concise text description"},
            {"name": "verbose", "description": "Verbose text description"},
        ],
        prompts=[
            "Generate {num_samples} examples of {label_name} ({label_description}) text in {language_name} about {topic}."
        ],
        expansion=PromptExpansionConfig(
            placeholders={"topic": ["technology", "health", "education", "environment"]},
            combinatorial=False,
            num_random_samples=3,
            max_samples=100,
        ),
        num_samples_per_prompt=5,
        languages={"en": "English"},
    )
    dataset = ClassificationDataset(config)
    # Each base prompt expands to 3 random samples: prompt_count=3
    # 2 classes * 3 prompts * 1 language * 5 samples * 3 providers = 90
    assert dataset.get_num_expected_rows(providers) == 90

def test_direct_utility_consistency(providers):
    # Test that the dataset method returns the same result as the direct utility function.
    config = ClassificationDatasetConfig(
        classes=[
            {"name": "concise", "description": "Concise text description"},
            {"name": "verbose", "description": "Verbose text description"},
        ],
        prompts=[
            "Generate {num_samples} examples of {label_name} ({label_description}) text in {language_name} about {topic}."
        ],
        expansion=PromptExpansionConfig(
            placeholders={"topic": ["technology", "health", "education", "environment"]},
            combinatorial=False,
            num_random_samples=3,
            max_samples=100,
        ),
        num_samples_per_prompt=5,
        languages={"en": "English"},
    )
    dataset = ClassificationDataset(config)
    assert dataset.get_num_expected_rows(providers) == utils._get_classification_num_expected_rows(config, providers)
