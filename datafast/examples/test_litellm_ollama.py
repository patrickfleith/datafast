
from ollama import chat
from pydantic import BaseModel

class Country(BaseModel):
  name: str
  capital: str
  languages: list[str]
  population_in_millions: float
  popular_traditional_cuisine: list[str]

response = chat(
  messages=[
    {
      'role': 'user',
      'content': 'Tell me about Canada.',
    }
  ],
  model='gemma3:4b',
  format=Country.model_json_schema(),
)

country = Country.model_validate_json(response.message.content)
print(country)

