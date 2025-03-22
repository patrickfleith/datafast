from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv("secrets.env")

class SimpleResponse(BaseModel):
    reasoning: str = Field(..., description="Reasoning of the answer")
    answer: str = Field(..., description="Answer to the question")


# client = InferenceClient(model="google/gemma-3-27b-it", api_key=os.getenv("HF_TOKEN"))
client = InferenceClient(model="Qwen/QwQ-32B", api_key=os.getenv("HF_TOKEN"))

resp = client.text_generation(
    prompt="Briefly explain how much is 3 x 5 x 7 x 2",
    max_new_tokens=2048,
    seed=42,
    grammar={"type": "json", "value": SimpleResponse.model_json_schema()},
    temperature=0.01,
)

# print(resp)
print(SimpleResponse.model_validate_json(resp))
