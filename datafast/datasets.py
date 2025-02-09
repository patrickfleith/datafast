from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any, Optional
from datasets import Dataset
from huggingface_hub import HfApi
from datafast.llms import LLMProvider
from datafast.prompts import classification_prompts
from datafast.schema.config import ClassificationConfig
from datafast.schema.data_rows import TextClassificationRow, LabelSource
from datafast.expanders import expand_prompts
import os


class DatasetBase(ABC):
    """Abstract base class for all dataset generators."""

    def __init__(self, config):
        self.config = config
        self.data_rows = []

    @abstractmethod
    def generate(self, llms=None):
        """Main method to generate the dataset."""
        pass

    def to_csv(self, filepath: str):
        """Convert self.data_rows to CSV."""
        raise NotImplementedError

    def to_parquet(self, filepath: str):
        """Convert self.data_rows to Parquet."""
        raise NotImplementedError

    def to_jsonl(self, filepath: str):
        """Convert self.data_rows to JSON lines."""
        self._save_rows(self.data_rows, filepath)

    def _save_rows(self, rows: list[Any], output_file: str):
        """Save rows to a file based on the file extension."""
        output_path = Path(output_file)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)

        if output_file.endswith(".jsonl"):
            with open(output_file, "w") as f:
                for row in rows:
                    f.write(row.model_dump_json() + "\n")
        else:
            raise ValueError(f"Unsupported output format: {output_file}")

    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
        train_size: Optional[float] = None,
        seed: Optional[int] = None,
        shuffle: Optional[bool] = True,
    ) -> str:
        """Push the dataset to Hugging Face Hub.

        Args:
            repo_id: The ID of the repository to push to (e.g., 'username/dataset-name')
            token: Hugging Face API token. If None, will look for token in environment
            private: Whether to create a private repository
            commit_message: Optional commit message. If None, uses a default message
            train_size: If provided, fraction of data to use for training
            (e.g., 0.8 for 80% train)
            seed: Optional random seed for train_test_split
            shuffle: Optional boolean to shuffle the data for train_test_split

        Returns:
            str: URL of the dataset on the Hub

        Raises:
            ValueError: If no data rows exist or if token is not provided
            ValueError: If invalid split parameters are provided
        """
        if not self.data_rows:
            raise ValueError("No data rows to push. Generate data first.")

        # Convert Pydantic models to dictionaries and handle UUID serialization
        data = []
        for row in self.data_rows:
            row_dict = row.model_dump()
            # Convert UUID to string
            if "uuid" in row_dict:
                row_dict["uuid"] = str(row_dict["uuid"])
            # Remove empty dictionaries that cause Parquet issues
            if "confidence_scores" in row_dict and not row_dict["confidence_scores"]:
                del row_dict["confidence_scores"]
            if "metadata" in row_dict and not row_dict["metadata"]:
                del row_dict["metadata"]
            data.append(row_dict)

        dataset = Dataset.from_list(data)

        # Create train/test split if requested
        if train_size is not None:
            if not 0 < train_size < 1:
                raise ValueError("train_size must be between 0 and 1")

            # Create the split
            splits = dataset.train_test_split(
                train_size=train_size,
                shuffle=shuffle,
                seed=seed,
            )
            dataset = splits  # splits is now a DatasetDict with 'train' and 'test' keys

        # Get token from env if not provided
        token = token or os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError(
                "No token provided and HF_TOKEN environment variable not set"
            )

        api = HfApi(token=token)

        # Create the repo if it doesn't exist
        api.create_repo(
            repo_id=repo_id, private=private, repo_type="dataset", exist_ok=True
        )

        # Push the dataset
        try:
            dataset.push_to_hub(
                repo_id,
                commit_message=commit_message or "Update dataset",
                token=token,
                private=private,
            )
        except Exception as e:
            if "did not recognize Python value type" in str(e):
                raise ValueError(
                    "Data type conversion error. Please ensure all fields "
                    "are of supported types. Original error: {str(e)}"
                )
            raise

        return f"https://huggingface.co/datasets/{repo_id}"


class TextEntries(BaseModel):
    entries: list[str] = Field(
        ..., description="List of generated texts for a specific class"
    )


class TextClassificationDataset(DatasetBase):
    def __init__(self, config: ClassificationConfig):
        super().__init__(config)
        self.config = config

    def generate(self, llms: list[LLMProvider]) -> "TextClassificationDataset":
        """Generate text classification data by calling multiple providers.

        Args:
            llms: List of LLM providers to use for generation. Must not be empty.

        Raises:
            ValueError: If no LLM providers are supplied or if no classes are defined.
        """
        if not llms:
            raise ValueError("At least one LLM provider must be supplied")

        if not self.config.classes:
            raise ValueError("No classification classes provided in config")

        # Get labels listing for context in prompts
        labels_listing = [label["name"] for label in self.config.classes]

        # Get languages from config, default to English if not specified
        languages = self.config.languages or {"en": "English"}

        # For each label, generate examples using all providers
        for label in self.config.classes:
            for lang_code, language_name in languages.items():
                # 1. Create base prompts for this label and language
                base_prompts = self.config.prompts or self._get_default_prompts()
                base_prompts = [
                    prompt.format(
                        num_samples=self.config.num_samples_per_prompt,
                        labels_listing=labels_listing,
                        label_name=label["name"],
                        label_description=label["description"],
                        language_name=language_name,
                    )  # Directly use language name
                    for prompt in base_prompts
                ]

                # 2. Expand prompts with configured variations
                expansions = expand_prompts(
                    prompt_templates=base_prompts, **self.config.expansion.model_dump()
                )

                # 3. For each expanded prompt, call each provider
                for expanded_prompt, meta in expansions:
                    for llm in llms:
                        try:
                            # Generate multiple examples using the LLM
                            response = llm.generate(
                                expanded_prompt, response_format=TextEntries
                            )

                            # Create a row for each generated example
                            for text in response.entries:
                                row = TextClassificationRow(
                                    text=text,
                                    label=label["name"],
                                    model_id=llm.model_id,
                                    label_source=LabelSource.SYNTHETIC,
                                    metadata={
                                        "language": lang_code
                                    },  # Store language info
                                )
                                self.data_rows.append(row)
                            print(f" Generated total of {len(self.data_rows)} examples")

                        except Exception as e:
                            print(f"Error with llm provider {llm.name}: {e}")

        # Final save at the end
        self.to_jsonl(self.config.output_file)
        return self

    def _get_default_prompts(self) -> list[str]:
        """Return the default prompt templates for text classification."""
        return classification_prompts.DEFAULT_TEMPLATES
