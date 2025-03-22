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
        model_id="meta-llama/Llama-3.3-70B-Instruct",  # You can change this to your preferred model
        api_key=os.getenv("HF_TOKEN")
    )

    # Example prompt for movie review analysis
    prompt = """
    Analyze this movie review and provide structured feedback:
    'Dune: Part Two exceeded all expectations! The stunning visuals and epic scale of 
    the desert battles were breathtaking. Timothée Chalamet and Zendaya delivered powerful 
    performances, and the score perfectly enhanced the intense atmosphere throughout.'
    """

    try:
        # Generate response with structured output
        response = provider.generate(prompt, MovieReview)
        # Print the structured response
        print("\nStructured Response:")
        print(response)

    except Exception as e:
        print(f"Error during generation: {str(e)}")


if __name__ == "__main__":
    load_dotenv("secrets.env")
    main()
