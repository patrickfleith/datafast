import os
from dotenv import load_dotenv
from pathlib import Path
import config
import instructor
import random
import uuid
import time
import os
from llm_utils import anthropic_generator, google_generator, openai_generator

# Load the environment variables from secrets.env file
env_path = Path(__file__).parents[3] / "secrets.env"
load_dotenv(dotenv_path=env_path)

# Setup your API in Colab Secrets and read it here. Pass it to genai to interact with Gemini.
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

from config import MODEL_IDS

labels_listing = [label['name'] for label in config.LABELS]

NUM_SAMPLES = 10
TEXT_WORDS_MIN = 30 
TEXT_WORDS_MAX = 50


from pydantic import BaseModel, Field

class TextEntries(BaseModel):
    # we can access the JSON Schema
    # TextEntries.model_json_schema()
    entries: list[str] = Field(
        ...,
        description="List of texts"
    )


def initialise_dataset(keys : list[str] = ["uuid", "text", "label", "model", "language"]) -> dict:
    return {key: [] for key in keys}


def get_generator_function(provider: str):
    """Get the appropriate generator function based on the provider."""
    generator_map = {
        'google': google_generator,
        'anthropic': anthropic_generator,
        'openai': openai_generator
    }
    return generator_map.get(provider)


dataset_dict = initialise_dataset()
print(dataset_dict)

rpm_counter = 0

for LABEL in config.LABELS:
    prompt = config.PROMPT_TEMPLATE.format(
        num_samples=NUM_SAMPLES,
        labels_listing=labels_listing,
        label_name=LABEL['name'],
        label_description=LABEL['description']
    ) + f" Your texts should be each between {TEXT_WORDS_MIN} and {TEXT_WORDS_MAX} words."

    for _, language in config.LANGUAGES.items():
        language_specific_prompt = prompt + f" Your answers must be in {language}."

        for model_config in MODEL_IDS:
            provider = model_config['provider']
            model_id = model_config['model_id']
            
            try:
                generator_func = get_generator_function(provider)
                if generator_func is None:
                    print(f"Unknown provider: {provider}")
                    continue

                # Call the appropriate generator with provider-specific arguments
                if provider == 'anthropic':
                    model_response = generator_func(
                        prompt=language_specific_prompt,
                        model_id=model_id,
                        temperature=0.3,
                        max_tokens=4096,
                        response_model=TextEntries
                    )
                else:
                    model_response = generator_func(
                        prompt=language_specific_prompt,
                        model_id=model_id,
                        response_model=TextEntries
                    )

                try:
                    for entry in model_response.entries:
                        dataset_dict['uuid'].append(str(uuid.uuid4()))
                        dataset_dict['text'].append(entry)
                        dataset_dict['label'].append(LABEL['label_id'])
                        dataset_dict['model'].append(model_id)
                        dataset_dict['language'].append(language)

                    rpm_counter += 1
                except Exception as e:
                    print(f"Error saving record in dataset dictionary: {str(e)}")

            except Exception as e:
                print(f"Error calling {provider} model {model_id}: {str(e)}")
                time.sleep(10)
                continue

            print(f"Generated a total of {rpm_counter*NUM_SAMPLES} instances so far")

        if rpm_counter >= 14:
            print("Quota pause...")
            time.sleep(60)
            rpm_counter = 0


from datasets import Dataset
ds = Dataset.from_dict(dataset_dict)

HF_TOKEN = os.getenv('HF_TOKEN')

HF_USERNAME="patrickfleith" # <--- replace with yours otherwise this won't work!
PRIVATE = False # choose if you want a private dataset. If false it will be public.

ds.push_to_hub(
    repo_id=f'{HF_USERNAME}/verbose-vs-concise-mini-demo',
    private=PRIVATE,
    token=HF_TOKEN
)

print(dataset_dict)
