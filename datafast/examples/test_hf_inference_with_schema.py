from pydantic import BaseModel
from pprint import pprint
from huggingface_hub import InferenceClient
import os

# Step 1: Define the Pydantic model
class ResponseFormatModel(BaseModel):
    reasoning: str
    answer: str

# Step 2: Generate the JSON Schema from the model
response_format_schema = ResponseFormatModel.model_json_schema()

# Step 3: Wrap it with the required "type" field for the LLM
response_format = {
    "type": "json",  # Explicitly specifying that this is a JSON Schema
    "value": response_format_schema,
}

# Print the response format and its type separately
print("Response Format Type:", type(response_format))
pprint(response_format)


client = InferenceClient(
	model="google/gemma-2-2b-it",
	api_key=os.getenv('HF_TOKEN')
)

messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

completion = client.chat.completions.create(
	messages=messages, 
	max_tokens=2048,
    response_format=response_format
)

print(completion.choices[0].message.content.strip())

