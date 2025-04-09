#!/usr/bin/env python
"""
Test script for LiteLLM providers in the datafast library.
This script demonstrates how to use the unified LiteLLM-based provider interface
for different LLM providers.
"""

from datafast.llms import create_provider, LLMProvider
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import sys
import argparse


# Load environment variables from .env file
load_dotenv("secrets.env")


# Structured response model for testing
class StructuredResponse(BaseModel):
    """A simple structured response model for testing LLM providers."""
    answer: str = Field(..., description="Direct answer to the question")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    tags: list[str] = Field(..., description="Tags or categories relevant to the answer")


def test_structured_generation(provider_instance, provider_name: str):
    """Test structured generation capability using a unified prompt."""
    print(f"\nTesting structured generation with {provider_name}...")
    
    try:
        # Simple question that requires structured thinking
        prompt = """What would happen if the moon was twice as close to Earth? 
        Provide a concise answer with scientific confidence level.
        Include relevant tags or categories for your answer. Answer in valid JSON format"""
        
        # Generate structured response
        response = provider_instance.generate(prompt, StructuredResponse)
        
        # Display results in a clean format
        print("\n📋 Structured Response:")
        print(f"📌 Answer: {response.answer}")
        print(f"🎯 Confidence: {response.confidence:.2f}")
        print(f"🏷️ Tags: {', '.join(response.tags)}")
        print("✅ Test successful!")
        
    except Exception as e:
        print(f"❌ Error testing with {provider_name}: {str(e)}")
        print(f"   Error type: {type(e).__name__}")


def test_raw_generation(provider_instance, provider_name: str):
    """Test raw text generation capability."""
    print(f"\nTesting raw text generation with {provider_name}...")
    
    try:
        # Simple question for raw text response
        prompt = "Explain quantum computing in 3 sentences."
        
        # Generate raw text response
        response = provider_instance.generate(prompt)
        
        # Display results
        print("\n📝 Raw Text Response:")
        print(f"{response}")
        print("✅ Test successful!")
        
    except Exception as e:
        print(f"❌ Error testing with {provider_name}: {str(e)}")
        print(f"   Error type: {type(e).__name__}")


def test_provider(name: str, model_id: Optional[str] = None, temperature: Optional[float] = None):
    """Test a specific provider with both structured and raw generation."""
    print(f"\n{'=' * 60}")
    print(f"🚀 Testing {name.upper()} provider with model: {model_id or 'default'}")
    print(f"Temperature: {temperature if temperature is not None else 'default'}")
    print("=" * 60)
    
    try:
        # Create provider instance
        provider = create_provider(
            provider=name,
            model_id=model_id,
            temperature=temperature
        )
        
        # Test both generation modes
        test_structured_generation(provider, name)
        test_raw_generation(provider, name)
        
    except Exception as e:
        print(f"❌ Failed to initialize {name} provider: {str(e)}")


def main():
    """Main function to run tests based on command line arguments."""
    parser = argparse.ArgumentParser(description="Test LiteLLM providers in datafast")
    parser.add_argument("--provider", type=str, default="openai", 
                        help="Provider to test (anthropic, google, openai, huggingface, ollama)")
    parser.add_argument("--model", type=str, default=None, 
                        help="Model ID to use (defaults to provider's default)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for generation (0.0 to 1.0)")
    parser.add_argument("--all", action="store_true", 
                        help="Test all available providers")
    
    args = parser.parse_args()
    
    # List of available providers
    providers = ["anthropic", "gemini", "openai"]
    
    if args.all:
        # Test all providers
        for provider in providers:
            try:
                test_provider(provider, args.model, args.temperature)
            except Exception as e:
                print(f"❌ Failed to test {provider}: {str(e)}")
    else:
        # Test specific provider
        test_provider(args.provider, args.model, args.temperature)


if __name__ == "__main__":
    main()
