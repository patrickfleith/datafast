# Core Concepts in Datafast

### Guiding Design Principle

> **Less Is More**: We seek data quality, data diversity and reliability over quantity. We don't measure success by shipping more features: we succeed if it works when you try it out.

This is a reason why we don't support a lot of LLM provider. We focus on what we know will work well for you.

### LLM Provider Abstraction

Datafast implements an LLM provider abstraction layer that decouples dataset generation logic from specific LLM implementations. This architecture enables:

- **Model Interchangeability**: Swap between different LLM providers without changing your dataset generation code
- **Multi-Model Generation**: Use multiple LLMs in parallel to increase output diversity and quality
- **Unified Interface**: Interact with all supported LLMs through a consistent API regardless of their underlying implementation differences

### Dataset Configuration

Datafast uses simple configuration objects to define your dataset requirements. Instead of writing complex code to set up your dataset, you just fill in a configuration with the details you need:

```python
# Example of a simple configuration for a classification dataset
config = ClassificationDatasetConfig(
    # Define what classes/categories you want
    classes=[
        {"name": "positive", "description": "Text expressing positive emotions"},
        {"name": "negative", "description": "Text expressing negative emotions"}
    ],
    # How many examples per prompt
    num_samples_per_prompt=5,
    # Where to save the result
    output_file="sentiment_dataset.jsonl",
    # What languages to generate
    languages={"en": "English", "fr": "French"}
)
```

This approach makes it easy to create, modify, and share dataset specifications without changing any underlying code.

## Architectural Components

### LLM Providers

The provider layer acts as an abstraction over different LLM APIs and services:

- **Integration Mechanism**: Utilizes [LiteLLM](https://github.com/BerriAI/litellm) for unified communication with various LLM services
- **Provider Classes**: Each supported provider (OpenAI, Anthropic, Gemini, Ollama) has a dedicated class handling authentication, request formatting, and response parsing
- **Structured Output**: Supports generation of structured data through Pydantic models for consistent and predictable output formats

### Dataset Classes

Each dataset type is implemented as a specialized class that:

1. **Processes Configuration**: Validates and prepares generation parameters
2. **Manages Prompt Creation**: Constructs effective prompts based on the dataset type and configuration
3. **Orchestrates Generation**: Orchestrates the LLM calls and data collection process
4. **Output Formatting**: Formats and saves the generated data in standardized formats

### Prompt Expansion System

The prompt expansion system is key and enables:

- **Combinatorial Variation**: Automatically creates multiple prompt variations through placeholder substitution
- **Diversity Control**: Provides mechanisms to balance between comprehensive coverage and targeted sampling
- **Efficiency**: Reduces manual prompt engineering effort while maximizing dataset diversity

## Conceptual Workflow

The datafast workflow follows a consistent pattern across all dataset types:

1. **Configuration**: Define the dataset parameters, classes/topics, and generation settings
2. **Logging Setup**: Configure logging to monitor the generation process (recommended)
3. **Prompt Design**: Create base prompts with mandatory and optional placeholders
4. **Provider Setup**: Initialize one or more LLM providers
5. **Generation**: Execute the generation process, which:
    - Expands prompts based on configuration
    - Distributes generation across providers
    - Collects and processes responses
6. **Output**: Save the resulting dataset to a file and optionally push to Hugging Face Hub

## Logging and Monitoring

Datafast includes comprehensive logging to provide visibility into the generation process:

### Why Configure Logging?

Without `configure_logger()`, your datafast scripts will run silently without:
- Progress indicators during generation
- Rate limiting warnings
- Success completion messages
- Detailed error information

### Basic Usage

```python
from datafast.logger_config import configure_logger

# Default: INFO level, console output with colors
configure_logger()

# With file logging for long-running jobs
configure_logger(level="INFO", log_file="generation.log")

# Debug mode for troubleshooting
configure_logger(level="DEBUG", log_file="debug.log")
```

## Dataset Diversity Mechanisms

Datafast employs multiple strategies to maximize dataset diversity:

### Multi-Dimensional Variation

Each dataset type supports some variations across multiple dimensions:

- **Content Dimensions**: Topics, document types, classes, personas, etc.
- **Linguistic Dimensions**: Multiple languages, writing styles, formality levels
- **Structural Dimensions**: Different formats, lengths, complexities

## Implementation Patterns

### Structured Data with Pydantic

Datafast uses Pydantic extensively for:

- **Configuration Validation**: Ensuring all required parameters are present and valid
- **Response Parsing**: Converting LLM text outputs into structured data objects
- **Type Safety**: Providing type hints throughout the codebase for better IDE support and error catching

### Asynchronous Generation

*Coming soon*

## Intended use cases

- Get initial evaluation text data instead of starting your LLM project blind.
- Increase diversity and coverage of another dataset by generating additional data.
- Experiment and test quickly LLM-based application PoCs
- Make your own datasets to fine-tune and evaluate language models for your application.

## Design Considerations

When working with datafast, consider these fundamental design aspects:

### Cost Efficiency

Synthetic data generation involves API costs. Datafast provides mechanisms to control costs:

- **Sampling**: Generate representative subsets rather than exhaustive combinations with combinatorial explosion
- **Provider Tiers**: Use less expensive models for initial testing, premium models for final datasets

### Customization

Datafast is designed to be flexible and customizable to meet specific requirements:

#### Custom Prompts

Each dataset type supports custom prompts that allow you to tailor the generation process:

```python
config = RawDatasetConfig(
    # Basic configuration
    document_types=["research paper", "blog post"],
    topics=["AI", "climate change"],
    num_samples_per_prompt=2,
    output_file="output.jsonl",
    languages={"en": "English"},
    
    # Custom prompt with placeholders
    prompts=[
        "Generate {num_samples} {document_type} in {language_name} " 
        "about {topic} with {{tone}} tone for {{audience}}."
    ]
)
```

#### Placeholder Systems

Datafast uses two types of placeholders for flexible prompt design:

- **Mandatory Placeholders** (`{like_this}`): Required by the dataset type and filled automatically
- **Optional Placeholders** (`{{like_this}}`): User-defined for creating prompt variations

#### Expansion Configuration

Control how your prompts are expanded with the `PromptExpansionConfig`:

```python
expansion_config = PromptExpansionConfig(
    placeholders={
        "tone": ["formal", "casual", "technical"],
        "audience": ["beginners", "experts", "general public"]
    },
    combinatorial=True,  # Generate all combinations
    # OR
    # combinatorial=False,  # Use random sampling instead
    # num_random_samples=100  # Number of random combinations
)
```

#### Multi-Provider Strategy

Use different LLM providers for different aspects of dataset generation:

```python
# Example for preference dataset generation
dataset.generate(
    question_gen_llm=OpenAIProvider(),
    chosen_response_gen_llm=AnthropicProvider(),
    rejected_response_gen_llm=GeminiProvider()
)
```

This approach allows you to leverage the strengths of different LLMs for specific generation tasks.