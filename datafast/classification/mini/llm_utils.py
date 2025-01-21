import anthropic
from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv
import instructor
import google.generativeai as genai
from openai import OpenAI



# Load the environment variables from secrets.env file
env_path = Path(__file__).parents[3] / "secrets.env"
load_dotenv(dotenv_path=env_path)

def get_messages(prompt: str) -> list[dict]:
    """Convert a single prompt into a message list format expected by LLM APIs.

    Args:
        prompt (str): The user's input prompt text

    Returns:
        list[dict]: A list of message dictionaries with system and user roles
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]


def anthropic_generator(prompt : str, model_id :str, response_model : BaseModel, max_tokens: int = 2056, temperature: float = 0.3):
    """Generate responses using Anthropic's Claude models with structured output.

    Args:
        prompt (str): The input prompt to send to the model
        model_id (str): The Anthropic model identifier (e.g., "claude-3-opus-20240229")
        response_model (BaseModel): Pydantic model defining the expected response structure
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
        ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
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


def google_generator(prompt : str, model_id :str, response_model : BaseModel):
    """Generate responses using Google's Gemini models with structured output.

    Args:
        prompt (str): The input prompt to send to the model
        model_id (str): The Gemini model identifier (e.g., "gemini-pro")
        response_model (BaseModel): Pydantic model defining the expected response structure

    Returns:
        BaseModel: Structured response matching the provided response_model

    Raises:
        ValueError: If API key is missing or client initialization fails
        RuntimeError: If model generation fails
        Exception: For unexpected errors during execution
    """
    try:
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
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


def openai_generator(prompt : str, model_id :str, response_model : BaseModel):
    
    openai_model = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
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

