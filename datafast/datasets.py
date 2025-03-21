from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any, Optional
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from datafast.llms import LLMProvider
from datafast.prompts import (
    classification_prompts,
    question_generation_prompts,
    mcq_prompts,
    text_prompts,
)
from datafast.schema.config import (
    ClassificationConfig,
    TextDatasetConfig,
    UltraChatDatasetConfig,
    MCQDatasetConfig,
)
from datafast.schema.data_rows import (
    ChatRow,
    TextClassificationRow,
    LabelSource,
    TextRow,
    TextSource,
    MCQRow,
    MCQSource,
)
from datafast.expanders import expand_prompts
import os


class TextEntries(BaseModel):
    entries: list[str] = Field(..., description="List of generated texts")


class QAEntry(BaseModel):
    question: str = Field(..., description="Question")
    answer: str = Field(..., description="Answer")

class QAEntries(BaseModel):
    entries: list[QAEntry] = Field(..., description="List of generated QAs")


class UserQuestions(BaseModel):
    questions: list[str] = Field(..., description="List of user questions")


class ReformulatedUserQuestion(BaseModel):
    question: str = Field(..., description="Reformulated user question")


class Answer(BaseModel):
    answer: str = Field(..., description="Answer to the user question")


class FollowupQuestion(BaseModel):
    question: str = Field(
        ..., description="Followup question of a user to an AI assistant response."
    )


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


class TextDataset(DatasetBase):
    def __init__(self, config: TextDatasetConfig):
        super().__init__(config)
        self.config = config

    def generate(self, llms: list[LLMProvider]) -> "TextDataset":
        """Generate text data by calling multiple providers.

        Args:
            llms: List of LLM providers to use for generation.

        Raises:
            ValueError: If no LLM providers are supplied or if text_attributes are missing.
        """
        if not llms:
            raise ValueError("At least one LLM provider must be supplied")

        # Get languages from config, default to English if not specified
        languages = self.config.languages or {"en": "English"}

        # For each language, generate examples using all providers
        for document_type in self.config.document_types:
            for topic in self.config.topics:
                for lang_code, language_name in languages.items():
                    # Add language to text attributes for prompt generation
                    # text_attrs = self.config.text_attributes.copy()
                    # text_attrs['language_name'] = language_name
                    # text_attrs['num_samples'] = str(self.config.num_samples_per_prompt)

                    # 1. Create base prompts for this language
                    base_prompts = self.config.prompts or self._get_default_prompts()
                    base_prompts = [
                        prompt.format(
                            num_samples=self.config.num_samples_per_prompt,
                            language_name=language_name,
                            document_type=document_type,
                            topic=topic,
                        )
                        for prompt in base_prompts
                    ]

                    # 2. Expand prompts with configured variations
                    expansions = expand_prompts(
                        prompt_templates=base_prompts,
                        **self.config.expansion.model_dump(),
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
                                    row = TextRow(
                                        text=text,
                                        text_source=TextSource.SYNTHETIC,
                                        model_id=llm.model_id,
                                        metadata={
                                            "language": lang_code,
                                            "document_type": document_type,
                                            "topic": topic,
                                        },
                                    )
                                    self.data_rows.append(row)
                                print(
                                    f" Generated total of {len(self.data_rows)} examples"
                                )

                            except Exception as e:
                                print(f"Error with llm provider {llm.name}: {e}")

        # Final save at the end
        self.to_jsonl(self.config.output_file)
        return self

    def _get_default_prompts(self) -> list[str]:
        """Return the default prompt templates for text generation."""
        return text_prompts.DEFAULT_TEMPLATES


class UltraChatDataset(DatasetBase):
    def __init__(self, config: UltraChatDatasetConfig):
        super().__init__(config)
        self.config = config

    def generate(self, llms: list[LLMProvider]) -> "TextDataset":
        if not llms:
            raise ValueError("At least one LLM provider must be supplied")

        # Get languages from config, default to English if not specified
        languages = self.config.languages or {"en": "English"}

        # For each language, generate examples using all providers
        for lang_code, language_name in languages.items():
            for topic, subtopics in self.config.topics_and_subtopics.items():
                for subtopic in subtopics:
                    # 1. Create base prompts for this language
                    base_prompts = (
                        self.config.question_generation_prompts
                        or self._get_default_question_generation_prompts()
                    )

                    base_prompts = [
                        prompt.format(
                            num_samples=self.config.num_samples,
                            language_name=language_name,
                            domain=self.config.domain,
                            topic=topic,
                            subtopic=subtopic,
                        )
                        for prompt in base_prompts
                    ]

                    # 2. Expand prompts with configured variations
                    expansions = expand_prompts(
                        prompt_templates=base_prompts,
                        **self.config.expansion.model_dump(),
                    )

                    # 3. For each expanded prompt, call each provider in UltraChat iteration
                    for i, (expanded_prompt, meta) in enumerate(expansions, 1):
                        for llm in llms:
                            try:
                                # Generate multiple examples using the LLM
                                # --- Here goes the ultraChat loop ---
                                opening_questions = llm.generate(
                                    expanded_prompt, response_format=UserQuestions
                                )

                                for opening_question in opening_questions.questions:
                                    random_persona = np.random.choice(
                                        self.config.personas
                                    )
                                    reformulation_prompt = self._get_default_persona_question_reformulation_prompt()
                                    reformulated_question = llm.generate(
                                        prompt=reformulation_prompt.format(
                                            question=opening_question,
                                            persona=random_persona,
                                            topic=topic,
                                            subtopic=subtopic,
                                        ),
                                        response_format=ReformulatedUserQuestion,
                                    )

                                    # simulate the assistant response to the opening question
                                    assistant_prompt = (
                                        self._get_default_simulated_assistant_prompt()
                                    )
                                    assistant_response = llm.generate(
                                        prompt=assistant_prompt.format(
                                            domain=self.config.domain,
                                            topic=topic,
                                            subtopic=subtopic,
                                            question=reformulated_question.question,
                                        ),
                                        response_format=Answer,
                                    )

                                    # choose to continue the conversation or not (proba 0.5)
                                    count = 1
                                    messages = [
                                        {
                                            "role": "user",
                                            "content": reformulated_question.question,
                                        },
                                        {
                                            "role": "assistant",
                                            "content": assistant_response.answer,
                                        },
                                    ]

                                    # assemble the dialog to prompt the user
                                    dialog_summary = f"{reformulated_question.question}\n{assistant_response.answer}"

                                    while (count < self.config.max_turns) and (
                                        np.random.random()
                                        < self.config.conversation_continuation_prob
                                    ):
                                        # simulate the user follow-up question
                                        followup_prompt = (
                                            self._get_default_user_followup_prompt()
                                        )
                                        followup_question = llm.generate(
                                            prompt=followup_prompt.format(
                                                dialog_summary=dialog_summary,
                                                persona=random_persona,
                                                subtopic=subtopic,
                                                domain=self.config.domain,
                                            ),
                                            response_format=ReformulatedUserQuestion,
                                        )
                                        # simulate the assistant response
                                        messages.append(
                                            {
                                                "role": "user",
                                                "content": followup_question.question,
                                            }
                                        )
                                        ai_response = llm.generate(
                                            messages=messages, response_format=Answer
                                        )

                                        dialog_summary += f"\n{followup_question.question}\n{ai_response.answer}"
                                        messages.append(
                                            {
                                                "role": "assistant",
                                                "content": ai_response.answer,
                                            }
                                        )

                                        count += 1
                                        if count >= self.config.max_turns:
                                            break

                                    # Create a row for each generated example
                                    row = ChatRow(
                                        opening_question=messages[0]["content"],
                                        messages=messages,
                                        model_id=llm.model_id,
                                        metadata={
                                            "language": lang_code,
                                            "domain": self.config.domain,
                                            "topic": topic,
                                            "subtopic": subtopic,
                                        },
                                        persona=random_persona,
                                    )
                                    self.data_rows.append(row)
                                print(
                                    f" Generated total of {len(self.data_rows)} examples"
                                )

                            except Exception as e:
                                import traceback
                                error_trace = traceback.format_exc()
                                print(f"\nError with llm provider {llm.name}:\n{error_trace}")
                                print(f"Error occurred at response type: {response_format.__name__ if 'response_format' in locals() else 'unknown'}")
                                if 'reformulated_question' in locals():
                                    print(f"Last reformulated_question: {reformulated_question}")

        self.to_jsonl(self.config.output_file)
        return self

    def _get_default_question_generation_prompts(self) -> list[str]:
        return question_generation_prompts.DOMAIN_TOPIC_SUBTOPIC_N_QUESTION_GENERATION_DEFAULT_TEMPLATES

    def _get_default_persona_question_reformulation_prompt(self) -> str:
        return (
            question_generation_prompts.PERSONA_QUESTION_REFORMULATION_DEFAULT_TEMPLATE
        )

    def _get_default_simulated_assistant_prompt(self) -> str:
        return question_generation_prompts.SIMULATED_ASSISTANT_DEFAULT_TEMPLATE

    # def _get_default_user_system_prompt(self) -> str:
    #     return question_generation_prompts.USER_SYSTEM_PROMPT_TEMPLATE

    def _get_default_user_followup_prompt(self) -> str:
        return question_generation_prompts.USER_FOLLOWUP_PROMPT_TEMPLATE


class MCQDataset(DatasetBase):
    def __init__(self, config: MCQDatasetConfig):
        super().__init__(config)
        self.config = config

    def generate(self, llms: list[LLMProvider]) -> "MCQDataset":
        """
        Generate multiple choice questions by calling providers for questions and then for incorrect answers.
        
        Args:
            llms: List of LLM providers to use for generation. Must not be empty.
            
        Raises:
            ValueError: If no LLM providers are supplied or if required configuration is missing.
        """
        if not llms:
            raise ValueError("At least one LLM provider must be supplied")
            
        # Load the dataset from Hugging Face
        try:
            hf_dataset = load_dataset(self.config.hf_dataset_name)
            # Most datasets have a 'train' split, but fallback to first available split
            split_names = list(hf_dataset.keys())
            if not split_names:
                raise ValueError(f"No splits found in dataset {self.config.hf_dataset_name}")
                
            main_split = "train" if "train" in split_names else split_names[0]
            dataset = hf_dataset[main_split]
            
            # Limit the number of samples if specified
            if self.config.sample_count is not None:
                dataset = dataset.select(range(min(self.config.sample_count, len(dataset))))
                
        except Exception as e:
            raise ValueError(f"Error loading dataset {self.config.hf_dataset_name}: {e}")
            
        # Get languages from config, default to English if not specified
        languages = self.config.languages or {"en": "English"}
        
        # For each document, generate questions and answers
        for sample in dataset:
            if self.config.text_column not in sample:
                print(f"Warning: Column {self.config.text_column} not found in sample, skipping")
                continue
                
            document = sample[self.config.text_column]
            if not document or len(document.strip()) < self.config.min_document_length:  # Skip very short documents
                continue
            if len(document.strip()) > self.config.max_document_length: # Skip very long documents
                continue
                
            for lang_code, language_name in languages.items():
                # 1. First call: Generate questions and correct answers
                question_prompts = self.config.prompts or self._get_default_prompts()
                question_prompts = [
                    prompt.format(
                        num_samples=self.config.num_samples_per_prompt,
                        language_name=language_name,
                        document=document,
                    )
                    for prompt in question_prompts
                ]
                
                # Expand prompts with configured variations
                question_expansions = expand_prompts(
                    prompt_templates=question_prompts,
                    **self.config.expansion.model_dump(),
                )
                
                # Process each expanded prompt
                for expanded_prompt, meta in question_expansions:
                    for llm in llms:
                        # Use the first LLM provider to generate questions and correct answers
                        try:
                            # Generate questions with the correct answers
                            response = llm.generate(expanded_prompt, response_format=QAEntries)
                            
                            for qa_entry in response.entries:
                                # Extract question and correct answer from the QAEntry
                                try:
                                    # QAEntry already has question and answer fields
                                    question_part = qa_entry.question
                                    correct_answer = qa_entry.answer
                                    
                                    # 2. Second call: Generate incorrect answers
                                    distractor_prompt = self.config.distractor_prompt or self._get_distractor_prompt().format(
                                        question=question_part,
                                        correct_answer=correct_answer,
                                        language_name=language_name,
                                    )
                                    
                                    try:
                                        # Use TextEntries for the distractor response since we need a list of incorrect answers
                                        distractor_response = llm.generate(
                                            distractor_prompt, response_format=TextEntries
                                        )
                                        
                                        # Parse the incorrect answers
                                        incorrect_answers = []
                                        for entry in distractor_response.entries:
                                            incorrect_answers.append(entry.strip())
                                        
                                        if len(incorrect_answers) >= 3:
                                            # Create MCQ row with the question, correct answer, and incorrect answers
                                            row = MCQRow(
                                                source_document=document,
                                                question=question_part,
                                                correct_answer=correct_answer,
                                                incorrect_answer_1=incorrect_answers[0],
                                                incorrect_answer_2=incorrect_answers[1],
                                                incorrect_answer_3=incorrect_answers[2],
                                                model_id=llm.model_id,
                                                mcq_source=MCQSource.SYNTHETIC,
                                                metadata={
                                                    "language": lang_code,
                                                    "source_dataset": self.config.hf_dataset_name,
                                                },
                                            )
                                            self.data_rows.append(row)
                                        else:
                                            print(f"Warning: Not enough incorrect answers generated (got {len(incorrect_answers)}, need 3)")
                                    except Exception as e:
                                        print(f"Error generating distractors: {e}")
                                except Exception as e:
                                    print(f"Error processing entry: {e}")
                            print(f" Generated total of {len(self.data_rows)} MCQs")
                        except Exception as e:
                            print(f"Error with llm provider {llm.name}: {e}")
        
        # Final save at the end
        self.to_jsonl(self.config.output_file)
        return self
    
    def _get_default_prompts(self) -> list[str]:
        """Return the default prompt templates for MCQ generation."""
        return mcq_prompts.DEFAULT_TEMPLATES
    
    def _get_distractor_prompt(self) -> str:
        """Return the prompt template for generating incorrect answers."""
        return mcq_prompts.DISTRACTOR_TEMPLATE
