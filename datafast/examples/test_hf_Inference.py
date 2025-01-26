from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv('secrets.env')

client = InferenceClient(
	model="google/gemma-2-2b-it",
	api_key=os.getenv('HF_TOKEN')
)

messages = [
	{
		"role": "user",
		"content": "What is the best cheese in France. Choose one."
	}
]

completion = client.chat.completions.create(
	messages=messages, 
	max_tokens=2048
)

print(completion.choices[0].message.content.strip())


