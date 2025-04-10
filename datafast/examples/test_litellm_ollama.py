#!/usr/bin/env python
"""
A super simple test script for Ollama using LiteLLM integration.

This script demonstrates how to use the datafast LLMProvider with Ollama models.
Requires Ollama to be installed and running locally: https://ollama.com/

Before running, make sure you have a model pulled with:
    ollama pull gemma3:12b
"""

from datafast.llms import create_provider
from pydantic import BaseModel, Field
import argparse


class SimpleAnswer(BaseModel):
    """A simple response model for structured output from Ollama models."""
    answer: str = Field(..., description="The direct answer to the question")
    explanation: str = Field(..., description="A brief explanation of the answer")


def test_ollama_simple(model_id: str = "gemma3:12b", temperature: float = 0.7):
    """Run a simple test with Ollama using LiteLLM integration."""
    print(f"\n{'=' * 60}")
    print(f"🦙 Testing Ollama with LiteLLM integration")
    print(f"📊 Model: {model_id}")
    print(f"🌡️ Temperature: {temperature}")
    print("=" * 60)
    
    # Create the Ollama provider
    provider = create_provider(
        provider="ollama",
        model_id=model_id,
        temperature=temperature
    )
    
    # Test with a simple prompt
    prompt = "What is the capital of Japan? Give a brief explanation."
    print(f"\n📝 Prompt: {prompt}")
    print("⏳ Generating response...")
    
    try:
        # Test raw text generation
        raw_response = provider.generate(prompt)
        print("\n📄 Raw Response:")
        print(f"{raw_response}")
        
        # Test structured response
        print("\n⏳ Generating structured response...")
        structured_prompt = """What is the capital of Japan? 
        Provide your answer in valid JSON format with 'answer' and 'explanation' fields."""
        
        structured_response = provider.generate(structured_prompt, SimpleAnswer)
        print("\n🧩 Structured Response:")
        print(f"📍 Answer: {structured_response.answer}")
        print(f"💡 Explanation: {structured_response.explanation}")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def main():
    """Parse command line arguments and run the test."""
    parser = argparse.ArgumentParser(description="Test Ollama with LiteLLM integration")
    parser.add_argument(
        "--model", 
        type=str, 
        default="gemma3:12b",
        help="Ollama model to use (default: gemma3:12b)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for generation (0.0 to 1.0, default: 0.7)"
    )
    
    args = parser.parse_args()
    test_ollama_simple(args.model, args.temperature)


if __name__ == "__main__":
    main()
