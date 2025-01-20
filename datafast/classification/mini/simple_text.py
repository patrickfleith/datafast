import os
from dotenv import load_dotenv
from pathlib import Path
import config
import instructor
import random
import uuid
import time
import os

# Load the environment variables from secrets.env file
env_path = Path(__file__).parents[3] / "secrets.env"
load_dotenv(dotenv_path=env_path)

# Setup your API in Colab Secrets and read it here. Pass it to genai to interact with Gemini.
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')


MODEL_ID = 'gemini-1.5-flash'

import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)
google_model = genai.GenerativeModel(model_name=MODEL_ID)


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


client = instructor.from_gemini(
    client=google_model,
    mode=instructor.Mode.GEMINI_JSON,
)


def initialise_dataset(keys : list[str] = ["uuid", "text", "label", "model", "language"]) -> dict:
    return {key: [] for key in keys}


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

        try:
            model_response = client.messages.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI Assistant"},
                    {
                        "role": "user",
                        "content": language_specific_prompt},
                ],
                response_model=TextEntries
            )

            try:
                for entry in model_response.entries:
                    dataset_dict['uuid'].append(str(uuid.uuid4()))  # Generate a unique ID
                    dataset_dict['text'].append(entry)
                    dataset_dict['label'].append(LABEL['label_id'])
                    dataset_dict['model'].append(MODEL_ID)
                    dataset_dict['language'].append(language)

                rpm_counter += 1
            except:
                print("Error saving record in dataset dictionary")

        except:
            print("Error calling model")
            time.sleep(10) # wait for 10s if an exception occured.
            pass

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
