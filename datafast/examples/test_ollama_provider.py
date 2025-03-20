from datafast.llms import create_provider
from pydantic import BaseModel, Field
from typing import Optional
import sys

"""
A simple test script for the OllamaProvider.
This script requires Ollama to be installed and running locally.

You can install Ollama from https://ollama.com/ and then run:
ollama pull gemma3:4b
"""


class SimpleResponse(BaseModel):
    """A simple response model with minimal fields to test the OllamaProvider."""
    answer: str = Field(..., description="The answer to the question")
    reasoning: str = Field(..., description="The reasoning behind the answer")


def test_ollama(model_id: str = "gemma3:4b"):
    """Test the OllamaProvider with a simple query."""
    print(f"\n{'=' * 50}")
    print(f"Testing Ollama provider with model {model_id}")
    print("=" * 50)

    try:
        # Create the provider
        provider = create_provider("ollama", model_id)
        
        # Simple test prompt
        prompt = "What is the capital of France? Provide a short answer and brief reasoning."
        
        print(f"Sending prompt: {prompt}")
        print("Waiting for response (this might take a bit)...")
        
        # Generate response
        response = provider.generate(prompt, SimpleResponse)
        
        print("\nResponse received:")
        print(f"Answer: {response.answer}")
        print(f"Reasoning: {response.reasoning}")
        print("\nTest successful!")
        
    except Exception as e:
        print(f"Error testing Ollama provider: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    # Get model from command line arguments if provided
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma3:4b"
    test_ollama(model)
