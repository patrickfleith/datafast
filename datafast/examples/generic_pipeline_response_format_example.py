"""Simple test for create_response_model function."""

from datafast.schema.config import GenericPipelineDatasetConfig
from datafast.utils import create_response_model
from datafast.logger_config import configure_logger

# Configure logger
configure_logger()

# Test with multiple columns and num_samples_per_prompt = 3
config = GenericPipelineDatasetConfig(
    hf_dataset_name="imdb",
    input_columns=["text"],
    output_columns=["summary", "sentiment"],
    prompts=["Analyze: {text}. Language: {language}. Generate {num_samples} responses."],
    num_samples_per_prompt=3
)

ResponseModel = create_response_model(config)

print(ResponseModel.model_json_schema())
