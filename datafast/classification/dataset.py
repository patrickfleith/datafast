import config
import templates
import google.generativeai as genai
from datasets import load_dataset
from pydantic import BaseModel
from dotenv import load_dotenv
import instructor
import random
import uuid
import time
import os

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


generation_prompt = templates.PROMPT_TEMPLATE.format(
    persona = "Leo, A small wood manufacturing business owner who just got his first kid.",
    num_entries = 2,
    style = config.STYLES['humorous'],
    context = config.CONTEXTS['night_time'],
    medium = config.MEDIUMS["social_media_post"],
    intent = config.INTENTS["request"],
    language=config.LANGUAGES['en']
)

print(generation_prompt)
print(GOOGLE_API_KEY)


genai.configure(api_key=GOOGLE_API_KEY)
