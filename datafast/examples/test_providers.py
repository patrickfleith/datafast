#!/usr/bin/env python3
"""Test the new provider-specific LLM implementation."""

import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field

# Add project root to path to make imports work
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider


class SimpleResponse(BaseModel):
    """Simple response model for testing structured output."""
    answer: str = Field(description="The answer to the question")
    reasoning: str = Field(description="The reasoning behind the answer")


def test_openai_provider():
    """Test the OpenAI provider."""
    print("Testing OpenAI provider...")
    
    # Create the provider
    provider = OpenAIProvider()
    
    # Test simple text generation
    prompt = "What is the capital of France? Keep it very brief."
    response = provider.generate(prompt=prompt)
    print(f"OpenAI text response: {response}")
    
    # Test structured output
    structured_prompt = "What is the capital of France? Answer with Paris and explain why."
    structured_response = provider.generate(
        prompt=structured_prompt, 
        response_format=SimpleResponse
    )
    print(f"OpenAI structured response: {structured_response}")
    print(f"Answer: {structured_response.answer}")
    print(f"Reasoning: {structured_response.reasoning}")
    
    # Test messages format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France? Keep it very brief."}
    ]
    messages_response = provider.generate(messages=messages)
    print(f"OpenAI messages response: {messages_response}")


def test_anthropic_provider():
    """Test the Anthropic provider."""
    print("\nTesting Anthropic provider...")
    
    # Create the provider
    provider = AnthropicProvider()
    
    # Test simple text generation
    prompt = "What is the capital of France? Keep it very brief."
    response = provider.generate(prompt=prompt)
    print(f"Anthropic text response: {response}")
    
    # Test structured output
    structured_prompt = "What is the capital of France? Answer with Paris and explain why."
    structured_response = provider.generate(
        prompt=structured_prompt, 
        response_format=SimpleResponse
    )
    print(f"Anthropic structured response: {structured_response}")
    print(f"Answer: {structured_response.answer}")
    print(f"Reasoning: {structured_response.reasoning}")
    
    # Test messages format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France? Keep it very brief."}
    ]
    messages_response = provider.generate(messages=messages)
    print(f"Anthropic messages response: {messages_response}")


def test_gemini_provider():
    """Test the Gemini provider."""
    print("\nTesting Gemini provider...")
    
    # Create the provider
    provider = GeminiProvider()
    
    # Test simple text generation
    prompt = "What is the capital of France? Keep it very brief."
    response = provider.generate(prompt=prompt)
    print(f"Gemini text response: {response}")
    
    # Test structured output
    structured_prompt = "What is the capital of France? Answer with Paris and explain why."
    structured_response = provider.generate(
        prompt=structured_prompt, 
        response_format=SimpleResponse
    )
    print(f"Gemini structured response: {structured_response}")
    print(f"Answer: {structured_response.answer}")
    print(f"Reasoning: {structured_response.reasoning}")
    
    # Test messages format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France? Keep it very brief."}
    ]
    messages_response = provider.generate(messages=messages)
    print(f"Gemini messages response: {messages_response}")


if __name__ == "__main__":
    # Run the tests - comment out the ones you don't want to run
    # You need to have the appropriate API keys set in your environment
    
    if os.getenv("OPENAI_API_KEY"):
        test_openai_provider()
    else:
        print("Skipping OpenAI test (OPENAI_API_KEY not set)")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        test_anthropic_provider()
    else:
        print("Skipping Anthropic test (ANTHROPIC_API_KEY not set)")
    
    if os.getenv("GEMINI_API_KEY"):
        test_gemini_provider()
    else:
        print("Skipping Gemini test (GEMINI_API_KEY not set)")
