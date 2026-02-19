# Datafast - Software Description

> **Version:** 0.0.33  
> **Author:** Patrick Fleith  
> **License:** Apache 2.0  
> **Python:** ≥3.10

---

## 1. Executive Summary

**Datafast** is a Python package for synthetic text dataset generation using Large Language Models (LLMs). It enables developers and researchers to quickly generate high-quality, diverse text datasets for training, fine-tuning, and evaluating language models.

**Core Value Proposition:**
- Generate text datasets in minutes instead of weeks
- Support for multiple dataset types (classification, MCQ, preference, instruction, raw text)
- Multi-provider LLM support (OpenAI, Anthropic, Gemini, Ollama, OpenRouter)
- Multi-lingual dataset generation
- Prompt expansion for maximizing diversity
- Direct Hugging Face Hub integration

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATAFAST PACKAGE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Configs    │───▶│   Datasets   │───▶│   Outputs    │                   │
│  │  (Pydantic)  │    │ (Generators) │    │ (JSONL/Hub)  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                                                │
│         │                   ▼                                                │
│         │           ┌──────────────┐                                         │
│         │           │ LLM Providers│                                         │
│         │           │  (LiteLLM)   │                                         │
│         │           └──────────────┘                                         │
│         │                   │                                                │
│         ▼                   ▼                                                │
│  ┌──────────────┐    ┌──────────────┐                                        │
│  │   Prompts    │───▶│  Expanders   │                                        │
│  │ (Templates)  │    │(Combinatorial│                                        │
│  └──────────────┘    │  /Random)    │                                        │
│                      └──────────────┘                                        │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Data Rows   │    │  Inspectors  │    │  Card Utils  │                   │
│  │  (Pydantic)  │    │   (Gradio)   │    │  (HF Cards)  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Structure

```
datafast/
├── __init__.py              # Package initialization, version, logger export
├── datasets.py              # Core dataset generators (1642 lines)
├── llms.py                  # LLM provider abstraction layer (729 lines)
├── schema/
│   ├── config.py            # Pydantic configuration models (674 lines)
│   └── data_rows.py         # Pydantic data row models (133 lines)
├── prompts/
│   ├── classification_prompts.py
│   ├── mcq_prompts.py
│   ├── preference_prompts.py
│   ├── question_generation_prompts.py
│   └── text_prompts.py
├── expanders.py             # Prompt expansion utilities (104 lines)
├── inspectors.py            # Gradio-based dataset visualization (379 lines)
├── utils.py                 # Shared utilities (391 lines)
├── card_utils.py            # Hugging Face dataset card utilities (177 lines)
├── llm_utils.py             # LLM helper functions (15 lines)
├── logger_config.py         # Loguru configuration (89 lines)
└── examples/                # Usage examples (15 files)
```

---

## 3. Core Components

### 3.1 Dataset Types

Datafast supports **6 dataset types**, each with dedicated configuration and generation logic:

| Dataset Type | Class | Config Class | Output Row Type | Use Case |
|-------------|-------|--------------|-----------------|----------|
| **Text Classification** | `ClassificationDataset` | `ClassificationDatasetConfig` | `TextClassificationRow` | Sentiment analysis, topic classification |
| **Raw Text** | `RawDataset` | `RawDatasetConfig` | `TextRow` | Document generation, content creation |
| **Instruction (Ultrachat)** | `UltrachatDataset` | `UltrachatDatasetConfig` | `ChatRow` | Conversational AI training |
| **Multiple Choice Questions** | `MCQDataset` | `MCQDatasetConfig` | `MCQRow` | Educational content, evaluation |
| **Preference** | `PreferenceDataset` | `PreferenceDatasetConfig` | `PreferenceRow` | RLHF training data |
| **Generic Pipeline** | `GenericPipelineDataset` | `GenericPipelineDatasetConfig` | Dynamic `GenericPipelineRow` | Custom transformations |

### 3.2 Dataset Base Class

All dataset types inherit from `DatasetBase`, which provides:

```python
class DatasetBase(ABC):
    def __init__(self, config):
        self.config = config
        self.data_rows = []

    @abstractmethod
    def generate(self, llms=None):
        """Main method to generate the dataset."""
        pass

    def inspect(self, random: bool = False) -> None:
        """Launch Gradio UI to browse generated data."""

    def to_jsonl(self, filepath: str, rows: list = None, append: bool = False):
        """Save rows to JSONL file."""

    def push_to_hub(self, repo_id: str, token: str = None, 
                    private: bool = True, train_size: float = None, ...) -> str:
        """Push dataset to Hugging Face Hub."""
```

**Key Features:**
- **Incremental saving:** Each batch is saved immediately after generation
- **Progress logging:** Real-time progress with provider/model info
- **Automatic dataset card:** Creates branded dataset card on Hub push
- **Train/test split:** Optional automatic splitting on push

---

## 4. Configuration System

### 4.1 Configuration Classes (Pydantic Models)

All configurations use **Pydantic BaseModel** with:
- Type validation
- Required placeholder validation in prompts
- Custom validators for business logic

#### 4.1.1 ClassificationDatasetConfig

```python
class ClassificationDatasetConfig(BaseModel):
    dataset_type: str = "text_classification"
    classes: list[dict[str, str]]           # [{"name": "...", "description": "..."}]
    prompts: list[str] | None = None        # Custom prompts (optional)
    num_samples_per_prompt: int = 5
    output_file: str = "classification.jsonl"
    expansion: PromptExpansionConfig = PromptExpansionConfig()
    languages: dict[str, str] = {"en": "English"}
```

**Required prompt placeholders:** `{num_samples}`, `{language_name}`, `{label_name}`, `{label_description}`

#### 4.1.2 RawDatasetConfig

```python
class RawDatasetConfig(BaseModel):
    dataset_type: str = "text"
    document_types: list[str]               # Required: e.g., ["blog post", "email"]
    topics: list[str]                       # Required: e.g., ["AI", "climate"]
    prompts: list[str] | None = None
    num_samples_per_prompt: int = 5
    output_file: str = "text.jsonl"
    expansion: PromptExpansionConfig
    languages: dict[str, str]
```

**Required prompt placeholders:** `{num_samples}`, `{language_name}`, `{document_type}`, `{topic}`

#### 4.1.3 UltrachatDatasetConfig

```python
class UltrachatDatasetConfig(BaseModel):
    dataset_type: str = "instruction_dataset"
    conversation_continuation_prob: float = 0.5  # Probability of follow-up
    max_turns: int = 1                           # Max conversation turns (1-10)
    domain: str = "Science, Technology..."
    topics_and_subtopics: dict[str, list[str]]   # {"Physics": ["Quantum", "Relativity"]}
    personas: list[str]                          # User personas
    num_samples: int = 10
    question_generation_prompts: list[str] | None
    persona_question_reformulation_prompt: str | None
    simulated_assistant_prompt: str | None
    user_followup_prompt: str | None
    output_file: str
    expansion: PromptExpansionConfig
    languages: dict[str, str]
```

**Generation Flow:**
1. Generate opening questions for topic/subtopic
2. Randomly assign persona
3. Reformulate question based on persona
4. Generate assistant response
5. Probabilistically continue conversation with follow-ups

#### 4.1.4 MCQDatasetConfig

```python
class MCQDatasetConfig(BaseModel):
    dataset_type: str = "mcq_dataset"
    hf_dataset_name: str | None = None       # HuggingFace dataset
    local_file_path: str | None = None       # OR local file (CSV, TXT, PARQUET, JSONL)
    text_column: str                         # Required: column with source text
    context_column: str | None = None        # Optional: contextual info column
    num_samples_per_prompt: int = 3
    sample_count: int | None = None          # Limit source samples
    min_document_length: int = 100
    max_document_length: int = 10000
    prompts: list[str] | None
    distractor_prompt: str | None
    output_file: str
    expansion: PromptExpansionConfig
    languages: dict[str, str]
```

**Two-stage generation:**
1. Generate questions + correct answers from source documents
2. Generate 3 plausible but incorrect distractors per question

#### 4.1.5 PreferenceDatasetConfig

```python
class PreferenceDatasetConfig(BaseModel):
    dataset_type: str = "preference_dataset"
    input_documents: list[str]               # Source documents for questions
    num_samples_per_prompt: int = 3
    question_generation_prompts: list[str] | None
    chosen_response_generation_prompt: str | None
    rejected_response_generation_prompt: str | None
    evol_instruct: bool = False              # Evolutionary refinement (not implemented)
    llm_as_judge: bool = False               # Use LLM to score responses
    evolution_prompt: str | None
    judge_prompt: str | None
    output_file: str
    languages: dict[str, str]
```

**Special `generate()` signature:**
```python
def generate(self, 
             question_gen_llm: LLMProvider,
             chosen_response_gen_llm: LLMProvider,
             rejected_response_gen_llm: LLMProvider,
             evolution_llm: LLMProvider = None,
             judge_llm: LLMProvider = None)
```

**LLM-as-Judge feature:** When enabled, both responses are scored 1-10, and if the "rejected" scores higher, they're swapped.

#### 4.1.6 GenericPipelineDatasetConfig

```python
class GenericPipelineDatasetConfig(BaseModel):
    dataset_type: str = "generic_pipeline_dataset"
    hf_dataset_name: str | None = None
    local_file_path: str | None = None
    input_columns: list[str]                 # Columns to use as prompt input
    forward_columns: list[str] | None        # Columns to copy to output
    output_columns: list[str] | None         # Generated output column names
    prompts: list[str]                       # Custom prompt templates
    num_samples_per_prompt: int = 1
    skip_function: Callable[[dict], bool] | None  # Row filter function
    sample_count: int | None
    output_file: str
    expansion: PromptExpansionConfig
    languages: dict[str, str]
```

**Required prompt placeholders:** `{num_samples}`, `{language}`, plus at least one `input_column`

**Dynamic model creation:** Uses `pydantic.create_model()` to dynamically create response and row models based on `output_columns`.

### 4.2 Prompt Expansion System

The `PromptExpansionConfig` enables systematic prompt variation:

```python
class PromptExpansionConfig(BaseModel):
    placeholders: dict[str, list[str]] = {}  # {{key}}: [values]
    combinatorial: bool = True               # All combinations vs random sampling
    num_random_samples: int = 1              # Only if combinatorial=False
    max_samples: int = 1000                  # Safety limit
```

**Placeholder syntax:**
- **Required:** `{placeholder}` - single braces, filled by config
- **Optional/Expansion:** `{{placeholder}}` - double braces, filled by expander

**Expansion modes:**
1. **Combinatorial (default):** Generate all possible combinations
2. **Random sampling:** Generate `num_random_samples` random combinations

**Example:**
```python
expansion = PromptExpansionConfig(
    placeholders={
        "context": ["hike review", "restaurant review"],
        "style": ["brief", "detailed"]
    },
    combinatorial=True  # → 4 expanded prompts
)
```

---

## 5. LLM Provider System

### 5.1 Provider Architecture

All providers inherit from `LLMProvider` abstract base class:

```python
class LLMProvider(ABC):
    def __init__(self, model_id: str, api_key: str | None = None,
                 temperature: float | None = None,
                 max_completion_tokens: int | None = None,
                 top_p: float | None = None,
                 frequency_penalty: float | None = None,
                 rpm_limit: int | None = None,
                 timeout: int | None = None):
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def env_key_name(self) -> str: ...

    def generate(self, prompt: str | list[str] | None = None,
                 messages: list | None = None,
                 response_format: Type[T] | None = None) -> str | T | list:
        """Generate text or structured output."""
```

### 5.2 Supported Providers

| Provider | Class | Env Variable | Default Model | Notes |
|----------|-------|--------------|---------------|-------|
| **OpenAI** | `OpenAIProvider` | `OPENAI_API_KEY` | `gpt-5-mini-2025-08-07` | Uses responses endpoint, supports `reasoning_effort` |
| **Anthropic** | `AnthropicProvider` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` | Standard completion |
| **Google Gemini** | `GeminiProvider` | `GEMINI_API_KEY` | `gemini-2.0-flash` | Full parameter support |
| **Ollama** | `OllamaProvider` | `OLLAMA_API_BASE` | `gemma3:4b` | Local LLM, no API key needed |
| **OpenRouter** | `OpenRouterProvider` | `OPENROUTER_API_KEY` | `openai/gpt-5-mini` | Multi-model gateway |

### 5.3 Key Features

**Unified Interface via LiteLLM:**
- All providers use LiteLLM under the hood for consistency
- Automatic batch processing where supported
- Rate limiting with `rpm_limit` parameter

**Structured Output:**
- Pass a Pydantic model to `response_format` parameter
- Automatic JSON instruction injection
- Code fence stripping
- Validation error handling with helpful logging

**Rate Limiting:**
```python
provider = GeminiProvider(model_id="gemini-2.0-flash", rpm_limit=15)
# Automatically waits when approaching 15 requests/minute
```

---

## 6. Data Row Models

### 6.1 Row Type Hierarchy

All row types use Pydantic models with common fields:

```python
class TextClassificationRow(BaseModel):
    text: str
    label: str | list[str] | list[int]
    model_id: str | None = None
    label_source: LabelSource = LabelSource.SYNTHETIC
    confidence_scores: dict[str, float] | None = {}
    language: str | None = None
    uuid: UUID = Field(default_factory=uuid4)
    metadata: dict[str, str] = {}

class TextRow(BaseModel):
    text: str
    text_source: TextSource = TextSource.SYNTHETIC
    model_id: str | None = None
    language: str | None = None
    uuid: UUID = Field(default_factory=uuid4)
    metadata: dict[str, str] = {}

class ChatRow(BaseModel):
    opening_question: str
    messages: list[dict[str, str]]  # [{"role": "user/assistant", "content": "..."}]
    model_id: str | None = None
    language: str | None = None
    uuid: UUID = Field(default_factory=uuid4)
    metadata: dict[str, str] = {}
    persona: str

class MCQRow(BaseModel):
    source_document: str
    question: str
    correct_answer: str
    incorrect_answer_1: str
    incorrect_answer_2: str
    incorrect_answer_3: str
    model_id: str | None = None
    mcq_source: MCQSource = MCQSource.SYNTHETIC
    language: str | None = None
    uuid: UUID = Field(default_factory=uuid4)
    metadata: dict[str, str] = {}

class PreferenceRow(BaseModel):
    input_document: str
    question: str
    chosen_response: str
    rejected_response: str
    preference_source: PreferenceSource = PreferenceSource.SYNTHETIC
    chosen_model_id: str | None = None
    rejected_model_id: str | None = None
    language: str | None = None
    chosen_response_score: int | None = None
    rejected_response_score: int | None = None
    chosen_response_assessment: str | None = None
    rejected_response_assessment: str | None = None
    uuid: UUID = Field(default_factory=uuid4)
    metadata: dict[str, str] = {}
```

### 6.2 Source Enums

```python
class LabelSource(str, Enum):
    SYNTHETIC = "synthetic"
    VERIFIED = "verified"
    HUMAN = "human"
    CONSENSUS = "consensus"
```

---

## 7. Default Prompt Templates

### 7.1 Classification Prompt

```
I need text examples in order to train a machine learning model to classify 
between the following classes {labels_listing}. Your task is to generate 
{num_samples} texts written in {language_name} which are diverse and 
representative of what could be encountered for the '{label_name}' class. 
{label_description}. Do not exagerate, and ensure that it a realistic text 
while it belongs to the described class.
```

### 7.2 MCQ Prompts

**Question Generation:**
```
You are an expert at creating exam questions. Your task is to come up with 
{num_samples} difficult multiple choice questions written in {language_name} 
in relation to the following document along with the correct answer...
```

**Distractor Generation:**
```
You are an expert in creating plausible but incorrect answers for multiple 
choice questions. For the following question, and correct answer, generate 
3 short, plausible but incorrect answers in {language_name}...
```

### 7.3 Ultrachat Prompts

**Question Generation:**
```
You are an expert in {domain}. Generate a series of {num_samples} questions 
in {language_name} about {subtopic} in the context of {topic}
```

**Persona Reformulation:**
```
Your task is to reformulate the following question so that it is plausible 
for the specified persona...
```

**Assistant Response:**
```
You are specialized in the domain of {domain} and in particular about {topic} 
and specifically about {subtopic}. You task to answer to inquiries that 
showcase your depth of knowledge...
```

### 7.4 Preference Prompts

**Question Generation (3 templates available):**
```
Based on the following document, generate {num_samples} clear and specific 
questions in {language_name} that would require detailed responses...
```

**Chosen Response:**
```
You are an expert AI assistant known for providing exceptionally helpful, 
accurate, and comprehensive responses. Given the document and question below, 
provide a concise and well-structured answer in {language_name}...
```

**Rejected Response:**
```
Provide a response in {language_name} to the following question based on 
the document...
```

**Judge Scoring:**
```
You are an expert evaluator assessing the quality of responses from an AI 
assistant to user queries. Rate the following response on a scale from 1 to 10...
```

---

## 8. Utilities

### 8.1 Dataset Loading (`utils.py`)

```python
def load_dataset_from_source(
    hf_dataset_name: str | None = None,
    local_file_path: str | None = None,
    sample_count: int | None = None,
    text_column: str = "text"
) -> list[dict]:
```

**Supported formats:**
- Hugging Face datasets
- CSV files
- TXT files (one entry per line)
- Parquet files
- JSONL/JSON files

### 8.2 Expected Row Calculation

Each dataset type has a `get_num_expected_rows()` method that calculates:

```
expected_rows = (
    num_llms ×
    num_languages ×
    num_samples_per_prompt ×
    dataset_specific_factors ×
    num_prompt_expansions
)
```

### 8.3 Dynamic Model Creation (`utils.py`)

For `GenericPipelineDataset`:

```python
def create_response_model(config) -> type[BaseModel]:
    """Creates dynamic Pydantic model for LLM response based on output_columns."""

def create_generic_pipeline_row_model(config) -> type[BaseModel]:
    """Creates dynamic row model with input, forward, and output columns."""
```

---

## 9. Inspection System

### 9.1 Gradio-Based Inspectors

Each dataset type has a specialized inspector in `inspectors.py`:

| Dataset | Inspector Class | Function |
|---------|-----------------|----------|
| Classification | `ClassificationInspector` | `inspect_classification_dataset()` |
| MCQ | `MCQInspector` | `inspect_mcq_dataset()` |
| Preference | `PreferenceInspector` | `inspect_preference_dataset()` |
| Raw | `RawInspector` | `inspect_raw_dataset()` |
| Ultrachat | `UltrachatInspector` | `inspect_ultrachat_dataset()` |

**Usage:**
```python
dataset = ClassificationDataset(config)
dataset.generate(providers)
dataset.inspect(random=True)  # Launch Gradio UI
```

**Features:**
- Previous/Next navigation
- Sequential or random ordering
- Field-specific display (text, labels, metadata, etc.)
- Conversation formatting for chat datasets

---

## 10. Hugging Face Hub Integration

### 10.1 Push to Hub

```python
dataset.push_to_hub(
    repo_id="username/dataset-name",
    token=None,              # Uses HF_TOKEN env var if None
    private=True,
    commit_message=None,
    train_size=0.8,          # Optional train/test split
    seed=42,
    shuffle=True,
    upload_card=True         # Auto-generate dataset card
)
```

### 10.2 Dataset Card Generation (`card_utils.py`)

Automatically generates a branded dataset card with:
- Datafast badge/logo
- Version information
- Preserved dataset_info metadata

**Template:**
```yaml
---
{{ card_data }}
{{ config_data }}
---
[<img src="https://raw.githubusercontent.com/patrickfleith/datafast/main/assets/datafast-badge-web.png"
     alt="Built with Datafast" />](https://github.com/patrickfleith/datafast)

# {{ pretty_name }}

This dataset was generated using Datafast (v{{ datafast_version }}), an open-source 
package to generate high-quality and diverse synthetic text datasets for LLMs.
```

---

## 11. Logging System

### 11.1 Configuration (`logger_config.py`)

Uses **loguru** with configurable output:

```python
from datafast import configure_logger

# Default: INFO level, console only
configure_logger()

# With file logging
configure_logger(level="DEBUG", log_file="datafast.log")

# Production: JSON format
configure_logger(level="WARNING", serialize=True, log_file="prod.log")
```

**Features:**
- Color-coded console output
- File rotation (10MB max, 1 week retention)
- Compressed log archives
- JSON serialization option

### 11.2 Progress Logging

Each generation step logs:
```
Generated and saved 25 examples total | Provider: openai | Model: gpt-4 | Duration: 3.2s
```

---

## 12. Dependencies

### 12.1 Core Dependencies

```toml
dependencies = [
    "datasets>=3.0",        # Hugging Face datasets
    "instructor",           # Structured LLM outputs
    "google-generativeai",  # Gemini API
    "python-dotenv",        # Environment variables
    "anthropic",            # Anthropic API
    "openai",               # OpenAI API
    "pydantic",             # Data validation
    "litellm",              # Unified LLM interface
    "gradio",               # Dataset inspection UI
    "loguru",               # Logging
]
```

### 12.2 Development Dependencies

```toml
[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "ruff>=0.9.0",
]
```

---

## 13. Usage Patterns

### 13.1 Basic Classification Dataset

```python
from datafast.datasets import ClassificationDataset
from datafast.schema.config import ClassificationDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, GeminiProvider
from dotenv import load_dotenv

load_dotenv()

config = ClassificationDatasetConfig(
    classes=[
        {"name": "positive", "description": "Expressing positive emotions"},
        {"name": "negative", "description": "Expressing negative emotions"}
    ],
    num_samples_per_prompt=5,
    output_file="sentiment.jsonl",
    languages={"en": "English", "fr": "French"},
    expansion=PromptExpansionConfig(
        placeholders={"context": ["product review", "movie review"]},
        combinatorial=True
    )
)

providers = [
    OpenAIProvider(model_id="gpt-5-mini-2025-08-07"),
    GeminiProvider(model_id="gemini-2.0-flash")
]

dataset = ClassificationDataset(config)
print(f"Expected rows: {dataset.get_num_expected_rows(providers)}")
dataset.generate(providers)
dataset.push_to_hub("username/sentiment-dataset", train_size=0.8)
```

### 13.2 MCQ Generation from Existing Dataset

```python
from datafast.datasets import MCQDataset
from datafast.schema.config import MCQDatasetConfig
from datafast.llms import OpenAIProvider

config = MCQDatasetConfig(
    hf_dataset_name="patrickfleith/space_engineering_texts",
    text_column="text",
    sample_count=100,
    num_samples_per_prompt=3,
    min_document_length=100,
    max_document_length=10000,
    output_file="mcq_dataset.jsonl"
)

providers = [OpenAIProvider(model_id="gpt-5-mini-2025-08-07")]
dataset = MCQDataset(config)
dataset.generate(providers)
```

### 13.3 Generic Pipeline for Custom Transformations

```python
from datafast.datasets import GenericPipelineDataset
from datafast.schema.config import GenericPipelineDatasetConfig
from datafast.llms import OpenAIProvider

config = GenericPipelineDatasetConfig(
    hf_dataset_name="patrickfleith/personas",
    input_columns=["persona"],
    forward_columns=["category"],
    output_columns=["tweet", "cv"],
    num_samples_per_prompt=2,
    prompts=[
        "Generate {num_samples} texts in {language}: a tweet and CV for: {persona}"
    ],
    output_file="personas_generated.jsonl"
)

dataset = GenericPipelineDataset(config)
dataset.generate([OpenAIProvider()])
```

---

## 14. Error Handling

### 14.1 Validation Errors

Pydantic validators catch configuration issues early:
- Missing required placeholders in prompts
- Invalid placeholder references in expansion config
- Invalid data source specifications (must have HF dataset OR local file)
- Out-of-range parameter values

### 14.2 Generation Resilience

- Individual LLM failures are logged and skipped
- Batch continues with remaining providers
- Each batch is saved immediately (no data loss on crash)
- JSON parsing failures log content preview for debugging

### 14.3 Rate Limiting

Built-in rate limiting prevents API quota exhaustion:
```python
provider = GeminiProvider(rpm_limit=15)  # Max 15 requests/minute
```

---

## 15. Future Roadmap (from TODO.md)

### Planned Features:
- **Samplers concept:** Systematic sampling across configs, models, and parameters
- **Pipeline composition:** Composable steps (generate → evolve → score → filter → deduplicate)
- **RAG datasets**
- **More personas integration**
- **Seed-based generation**
- **More instruction dataset types**
- **Additional LLM providers**
- **Deduplication and filtering**
- **Enhanced dataset cards**

### Conceptual Direction:
Moving toward a more flexible pipeline architecture where:
1. Seeds/interim data flows through pipeline steps
2. Generation configs (prompts, LLMs, parameters) can be sampled
3. Steps can branch and merge
4. Mix of LLM-based and function-based steps

---

## 16. Testing

### 16.1 Test Structure

```
tests/
├── test_anthropic.py       # Anthropic provider tests
├── test_gemini.py          # Gemini provider tests
├── test_ollama.py          # Ollama provider tests
├── test_openai.py          # OpenAI provider tests
├── test_openrouter.py      # OpenRouter provider tests
├── test_expand_prompt.py   # Prompt expansion tests
├── test_get_num_expected_rows.py  # Row calculation tests
└── test_schemas.py         # Pydantic schema validation tests
```

### 16.2 Running Tests

```bash
pytest  # Requires API keys for LLM tests, Ollama server running
```

---

## 17. Summary

**Datafast** is a well-structured, modular package for synthetic text dataset generation that:

1. **Abstracts LLM complexity** through a unified provider interface
2. **Maximizes diversity** through prompt expansion and multi-provider support
3. **Ensures data quality** through Pydantic validation and structured outputs
4. **Integrates seamlessly** with the Hugging Face ecosystem
5. **Provides visibility** through Gradio-based inspection and comprehensive logging

The architecture follows a clear separation of concerns:
- **Configuration** (Pydantic models) → **Generation** (Dataset classes) → **Output** (JSONL/Hub)
- **Prompts** (Templates) → **Expansion** (Combinatorial/Random) → **LLM calls** (Providers)

This design makes it easy to:
- Add new dataset types
- Support new LLM providers
- Customize prompt templates
- Extend functionality through the generic pipeline
