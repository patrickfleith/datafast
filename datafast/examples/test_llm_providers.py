from datafast.llms import create_provider
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv


# Unified test response model
class StructuredResponse(BaseModel):
    """A simple structured response model for testing LLM providers."""
    answer: str = Field(..., description="Direct answer to the question")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    tags: list[str] = Field(..., description="Tags or categories relevant to the answer")


def test_structured_generation(provider_instance, provider_name: str):
    """Test structured generation capability using a unified prompt."""
    print(f"\nTesting structured generation with {provider_name}...")
    
    try:
        # Simple question that requires structured thinking
        prompt = """What would happen if the moon was twice as close to Earth? 
        Provide a concise answer scientific confidence level.
        Include relevant tags or categories for your answer."""
        
        # Generate structured response
        response = provider_instance.generate(prompt, StructuredResponse)
        
        # Display results in a clean format
        print("\n📋 Structured Response:")
        print(f"📌 Answer: {response.answer}")
        print(f"🎯 Confidence: {response.confidence:.2f}")
        print(f"🏷️ Tags: {', '.join(response.tags)}")
        print("✅ Test successful!")
        
    except NotImplementedError as e:
        print(f"⚠️ Feature not implemented for {provider_name}: {str(e)}")
        
    except Exception as e:
        print(f"❌ Error testing with {provider_name}: {str(e)}")
        print(f"   Error type: {type(e).__name__}")


def test_provider(name: str, model_id: Optional[str] = None):
    """Test a specific provider with structured generation."""
    print(f"\n{'=' * 60}")
    print(f"🚀 Testing {name.upper()} provider with model: {model_id or 'default'}")
    print("=" * 60)

    try:
        # Create the provider
        provider = create_provider(name, model_id)
        
        # Run the test
        test_structured_generation(provider, name)
        
    except Exception as e:
        print(f"❌ Error setting up {name} provider: {str(e)}")


def main():
    load_dotenv("secrets.env")

    # List of providers to test with their models
    providers = [
        # ("anthropic", "claude-3-5-haiku-latest"),
        # ("google", "gemini-1.5-flash"),
        # ("openai", "gpt-4o-mini"),
        ("ollama", "gemma3:4b"),
        # ("huggingface", "meta-llama/Llama-3.3-70B-Instruct"),
    ]
    
    print("🧪 STRUCTURED OUTPUT GENERATION TEST")
    print("Testing all providers with a simple structured generation task")
    
    for provider_name, model_id in providers:
        try:
            test_provider(provider_name, model_id)
        except Exception as e:
            print(f"\n{'=' * 60}")
            print(f"❌ Error testing {provider_name} provider with model {model_id}:")
            print(f"{str(e)}")
            print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
