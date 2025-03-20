from datafast.llms import create_provider
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv


# Simple sentiment response model
class SimpleResponse(BaseModel):
    sentiment: str = Field(..., description="Sentiment: positive, negative, or neutral")
    explanation: str = Field(..., description="Explanation of the sentiment")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")


# Complex response model for code analysis
class CodeIssue(BaseModel):
    severity: str = Field(..., description="Severity level: low, medium, high")
    description: str = Field(..., description="Description of the issue")
    suggestion: str = Field(..., description="Suggested fix")


class CodeAnalysis(BaseModel):
    language_detected: str = Field(..., description="Language detected in the code")
    main_purpose: str = Field(..., description="Main purpose of the code")
    potential_issues: list[CodeIssue] = Field(
        ..., description="Potential issues identified"
    )
    complexity_score: float = Field(
        ..., ge=0, le=10, description="Complexity score between 0.0 and 10.0"
    )
    suggested_improvements: list[str] = Field(
        ...,
        description="Suggested \
        improvements",
    )


def test_sentiment(provider_instance, provider_name: str):
    """Test sentiment analysis capability."""
    print("\nTesting sentiment analysis...")
    try:
        prompt = "Analyze the sentiment of this text: 'I absolutely love this \
            product! It's amazing!'"
        response = provider_instance.generate(prompt, SimpleResponse)

        print("\nSentiment Analysis Response:")
        print(f"Sentiment: {response.sentiment}")
        print(f"Explanation: {response.explanation}")
        print(f"Confidence: {response.confidence:.2f}")
    except Exception as e:
        print(f"Error testing sentiment analysis: {str(e)}")


def test_code_analysis(provider_instance, provider_name: str):
    """Test code analysis capability."""
    print("\nTesting code analysis...")
    try:
        code_snippet = """
def calculate_factorial(n):
    result = 1
    # Calculate factorial using a loop
    for i in range(1, n + 1):
        result = result * i
    return result

# Get user input
num = int(input("Enter a number: "))
if num >= 0:
    # Calculate factorial
    fact = calculate_factorial(num)
    print(f"Factorial of {num} is {fact}")
"""
        prompt = f"""Analyze this Python code and provide structured feedback:

{code_snippet}

Provide a detailed analysis including:
1. The main purpose of the code
2. Potential issues or bugs
3. Complexity assessment
4. Suggested improvements

Focus on both functionality and best practices."""

        response = provider_instance.generate(prompt, CodeAnalysis)

        print("\nCode Analysis Response:")
        print(f"Language: {response.language_detected}")
        print(f"Main Purpose: {response.main_purpose}")
        print("\nPotential Issues:")
        for issue in response.potential_issues:
            print(f"- [{issue.severity}] {issue.description}")
            print(f"  Suggestion: {issue.suggestion}")
        print(f"\nComplexity Score: {response.complexity_score}/10")
        print("\nSuggested Improvements:")
        for improvement in response.suggested_improvements:
            print(f"- {improvement}")
    except Exception as e:
        print(f"Error testing code analysis: {str(e)}")


def test_provider(name: str, model_id: Optional[str] = None):
    """Test a specific provider with multiple capabilities."""
    print(f"\n{'=' * 50}")
    print(f"Testing {name} provider...")
    print(f"Using model: {model_id or 'default'}")
    print("=" * 50)

    try:
        # Create the provider
        provider = create_provider(name, model_id)

        # Test both capabilities
        test_sentiment(provider, name)
        test_code_analysis(provider, name)

    except Exception as e:
        print(f"Error setting up {name} provider: {str(e)}")


def main():
    load_dotenv("secrets.env")

    # Test Anthropic (Claude)
    test_provider("anthropic", "claude-3-5-haiku-latest")

    # Test Google (Gemini)
    test_provider("google", "gemini-1.5-flash")

    # Test OpenAI (GPT-4)
    test_provider("openai", "gpt-4o-mini")
    
    # Test Ollama (local LLM)
    test_provider("ollama", "gemma3:4b")


if __name__ == "__main__":
    main()
