from abc import ABC, abstractmethod
from random import random, choice
from uuid import uuid4
from pydantic import BaseModel, Field
from pathlib import Path
import json
from typing import Any

from datafast.llms import LLMProvider
from datafast.prompts import classification_prompts
from datafast.schema.config import ClassificationConfig
from datafast.schema.data_rows import TextClassificationRow, LabelSource
from datafast.expanders import expand_prompts


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
            
        if output_file.endswith('.jsonl'):
            with open(output_file, 'w') as f:
                for row in rows:
                    f.write(row.model_dump_json() + '\n')
        else:
            raise ValueError(f"Unsupported output format: {output_file}")


class TextEntries(BaseModel):
    entries: list[str] = Field(
        ...,
        description="List of generated texts for a specific class"
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
        labels_listing = [label['name'] for label in self.config.classes]
        
        # For each label, generate examples using all providers
        for label in self.config.classes:
            # 1. Create base prompt for this label
            base_prompts = self.config.prompts or self._get_default_prompts()
            base_prompts = [
                prompt.format(
                    num_samples=self.config.num_samples_per_prompt,
                    labels_listing=labels_listing,
                    label_name=label["name"],
                    label_description=label["description"])
                for prompt in base_prompts
            ]
            
            # 2. Expand prompts
            expansions = expand_prompts(
                prompt_templates=base_prompts,
                **self.config.expansion.model_dump()
            )

            # 3. For each expanded prompt, call each provider
            for expanded_prompt, meta in expansions:
                for llm in llms:
                    try:
                        # Generate multiple examples using the LLM
                        response = llm.generate(
                            expanded_prompt,
                            response_format=TextEntries
                        )
                        
                        # Create a row for each generated example
                        for text in response.entries:
                            row = TextClassificationRow(
                                text=text,
                                label=label["name"],
                                model_id=llm.model_id,
                                label_source=LabelSource.SYNTHETIC,
                            )
                            self.data_rows.append(row)
                            
                    except Exception as e:
                        print(f"Error with llm provider {llm.name}: {e}")
        
        # Final save at the end
        self.to_jsonl(self.config.output_file)
        return self


    def _get_default_prompts(self) -> list[str]:
        """Return the default prompt templates for text classification."""
        return classification_prompts.DEFAULT_TEMPLATES
