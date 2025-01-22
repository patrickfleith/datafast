from llm_utils import get_messages
import anthropic
from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv
import instructor
import google.generativeai as genai
from openai import OpenAI

def anthropic_generator(prompt : str, model_id :str, response_model : BaseModel, api_key: str = None, max_tokens: int = 2056, temperature: float = 0.3):
    """Generate responses using Anthropic's Claude models with structured output.

    Args:
        prompt (str): The input prompt to send to the model
        model_id (str): The Anthropic model identifier (e.g., "claude-3-opus-20240229")
        response_model (BaseModel): Pydantic model defining the expected response structure
        api_key (str): Anthropic API key (optional)
        max_tokens (int, optional): Maximum number of tokens in the response. Defaults to 2056.
        temperature (float, optional): Sampling temperature. Defaults to 0.3.

    Returns:
        BaseModel: Structured response matching the provided response_model

    Raises:
        ValueError: If API key is missing or client initialization fails
        RuntimeError: If model generation fails
        Exception: For unexpected errors during execution
    """
    try:
        ANTHROPIC_API_KEY = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        try:
            anthropic_model = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            anthropic_client = instructor.from_anthropic(anthropic_model, mode=instructor.Mode.ANTHROPIC_TOOLS)
        except Exception as e:
            raise ValueError(f"Error initializing Anthropic client: {str(e)}")

        try:
            model_response = anthropic_client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=get_messages(prompt),
                temperature=temperature,
                response_model=response_model,
            )
            return model_response
        except Exception as e:
            raise RuntimeError(f"Error generating response: {str(e)}")

    except ValueError as ve:
        # Handle configuration and validation errors
        raise ve
    except RuntimeError as re:
        # Handle model generation errors
        raise re
    except Exception as e:
        # Handle any unexpected errors
        raise RuntimeError(f"Unexpected error in anthropic_generator: {str(e)}")


def google_generator(prompt : str, model_id :str, response_model : BaseModel, api_key: str = None):
    """Generate responses using Google's Gemini models with structured output.

    Args:
        prompt (str): The input prompt to send to the model
        model_id (str): The Gemini model identifier (e.g., "gemini-pro")
        response_model (BaseModel): Pydantic model defining the expected response structure
        api_key (str): Gemini API key (optional)

    Returns:
        BaseModel: Structured response matching the provided response_model

    Raises:
        ValueError: If API key is missing or client initialization fails
        RuntimeError: If model generation fails
        Exception: For unexpected errors during execution
    """
    try:
        GOOGLE_API_KEY = api_key or os.getenv('GOOGLE_API_KEY')
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        genai.configure(api_key=GOOGLE_API_KEY)
        
        try:
            google_model = genai.GenerativeModel(model_name=model_id)
        except Exception as e:
            raise ValueError(f"Invalid model ID or model initialization error: {str(e)}")

        google_client = instructor.from_gemini(
            client=google_model,
            mode=instructor.Mode.GEMINI_JSON
        )

        try:
            model_response = google_client.messages.create(
                messages=get_messages(prompt),
                response_model=response_model,
            )
            return model_response
        except Exception as e:
            raise RuntimeError(f"Error generating response: {str(e)}")

    except ValueError as ve:
        # Handle configuration and validation errors
        raise ve
    except RuntimeError as re:
        # Handle model generation errors
        raise re
    except Exception as e:
        # Handle any unexpected errors
        raise RuntimeError(f"Unexpected error in google_generator: {str(e)}")


def openai_generator(prompt : str, model_id :str, response_model : BaseModel, api_key: str = None):
    """Generate responses using OpenAI models with structured output.

    Args:
        prompt (str): The input prompt to send to the model
        model_id (str): The OpenAI model identifier (e.g., "gpt-4")
        response_model (BaseModel): Pydantic model defining the expected response structure
        api_key (str): OpenAI API key (optional)

    Returns:
        BaseModel: Structured response matching the provided response_model

    Raises:
        ValueError: If API key is missing or client initialization fails
        RuntimeError: If model generation fails
        Exception: For unexpected errors during execution
    """
    
    openai_model = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
    
    openai_client = instructor.from_openai(
        client=openai_model,
        mode=instructor.Mode.JSON
    )
    
    model_response = openai_client.chat.completions.create(
        model=model_id,
        messages=get_messages(prompt),
        response_model=response_model,
    )

    return model_response
