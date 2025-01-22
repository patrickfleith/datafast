from abc import ABC, abstractmethod
from typing import List

from datafast.providers.base import LLMProvider
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
        raise NotImplementedError


class TextClassificationDataset(DatasetBase):
    def __init__(self, config: ClassificationConfig):
        super().__init__(config)
        self.config = config

    def generate(self, llms: List[LLMProvider]) -> "TextClassificationDataset":
        """Generate text classification data by calling multiple providers.
        
        Args:
            llms: List of LLM providers to use for generation. Must not be empty.
            
        Raises:
            ValueError: If no LLM providers are supplied.
        """
        if not llms:
            raise ValueError("At least one LLM provider must be supplied")

        # 1. Gather the prompt templates
        prompt_templates = self.config.prompts or self._get_default_prompts()

        # 2. Expand placeholders
        expansions = expand_prompts(
            prompt_templates=prompt_templates,
            placeholders=self.config.expansion.placeholders,
            combinatorial=self.config.expansion.combinatorial,
            num_samples=self.config.expansion.num_samples
        )

        # 3. For each expanded prompt, call each provider
        for expanded_prompt, meta in expansions:
            for llm in llms:
                try:
                    response_text = llm.generate(expanded_prompt)
                    labels = []  # Assuming labels are empty for now
                    row = TextClassificationRow(
                        text=response_text,
                        labels=labels,
                        model_id=llm.model_id,
                        metadata=meta,
                        label_source=LabelSource.SYNTHETIC
                    )
                    self.data_rows.append(row)
                    
                except Exception as e:
                    # Log or handle errors, skip
                    print(f"Error with llm provider {llm.name}: {e}")
        
        # Final save at the end
        final_save(self.data_rows, self.config.output_file)
        return self


    def _get_default_prompts(self) -> List[str]:
        """Return the default prompt templates for text classification."""
        return classification_prompts.DEFAULT_TEMPLATES