from datafast.llms import HuggingFaceProvider
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import json

# Define a response schema for structured output
class MovieReview(BaseModel):
    sentiment: str
    explanation: str
    rating: int = Field(..., description="Rating out of 10")


def main():
    # Initialize the HuggingFace provider
    provider = HuggingFaceProvider(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # You can change this to your preferred model
        api_key=os.getenv("HF_TOKEN")
    )

    # Example prompt for movie review analysis
    prompt = """
    Analyze this movie review and provide structured feedback:
    'The new Spider-Man movie was amazing! The action scenes and the chemistry between 
    Tom Holland and Zendaya were so much fun to watch. The plot twists kept me on the 
    edge of my seat, though the special effects were somewhat over the top.'
    """

    try:
        # Generate response with structured output
        response = provider.generate(prompt, MovieReview)
        # Print the structured response
        print("\nStructured Response:")
        print(MovieReview.model_validate(json.loads(response.choices[0].message.content.strip())))

    except Exception as e:
        print(f"Error during generation: {str(e)}")


if __name__ == "__main__":
    load_dotenv("secrets.env")
    main()
