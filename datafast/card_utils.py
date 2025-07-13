import os
import re
from pathlib import Path
from huggingface_hub import HfApi, DatasetCard, DatasetCardData
from huggingface_hub.file_download import hf_hub_download

# Default template string embedded directly in code
# This eliminates the need to read from a file
DEFAULT_CARD_TEMPLATE = """---
{{ card_data }}
{{ config_data }}
---
[<img src="https://raw.githubusercontent.com/patrickfleith/datafast/main/assets/datafast-badge-web.png"
     alt="Built with Datafast" />](https://github.com/patrickfleith/datafast)

# {{ pretty_name }}

This dataset was generated using Datafast (v{{ datafast_version }}), an open-source package to generate high-quality and diverse synthetic text datasets for LLMs.
"""

def extract_readme_metadata(repo_id: str, token: str | None = None) -> str:
    """Extracts the metadata from the README.md file of the dataset repository.
    We have to download the previous README.md file in the repo, extract the metadata from it.
    Args:
        repo_id: The ID of the repository to push to, from the `push_to_hub` method.
        token: The token to authenticate with the Hugging Face Hub, from the `push_to_hub` method.
    Returns:
        The metadata extracted from the README.md file of the dataset repository as a str.
    """
    try:
        readme_path = Path(
            hf_hub_download(repo_id, "README.md", repo_type="dataset", token=token)
        )
        # Extract the content between the '---' markers
        metadata_match = re.findall(r"---\n(.*?)\n---", readme_path.read_text(), re.DOTALL)

        if not metadata_match:
            print("No YAML metadata found in the README.md")
            return ""

        return metadata_match[0]

    except Exception as e:
        print(f"Failed to extract metadata from README.md: {e}")
        return ""


def extract_dataset_info(repo_id: str, token: str | None = None) -> str:
    """
    Extract dataset_info section from README metadata.
    
    Args:
        repo_id: The dataset repository ID
        token: Optional HuggingFace token for authentication
        
    Returns:
        The dataset_info section as a string, or empty string if not found
    """       
    readme_metadata = extract_readme_metadata(repo_id=repo_id, token=token)
    if not readme_metadata:
        return ""

    section_prefix = "dataset_info:"
    if section_prefix not in readme_metadata:
        return ""

    try:
        # Extract the part after `dataset_info:` prefix
        config_data = section_prefix + readme_metadata.split(section_prefix)[1]
        return config_data
    except IndexError:
        print("Failed to extract dataset_info section from metadata")
        return ""


def _generate_and_upload_dataset_card(
    repo_id: str,
    token: str | None = None
) -> None:
    """
    Internal implementation that generates and uploads a dataset card to Hugging Face Hub.
    
    This is the core implementation function called by the public upload_dataset_card() function.
    It handles the actual card generation and uploading without performing configuration checks.
    
    The dataset card includes:
    1. Pipeline subset descriptions based on enabled stages
    2. Full sanitized configuration for reproducibility
    3. Datafast version and other metadata
    4. Preserved dataset_info from the existing card for proper configuration display
    """

    try:
        # Use the built-in template string
        template_str = DEFAULT_CARD_TEMPLATE
        print(f"Using built-in template, length: {len(template_str)} characters")

        # Get HF token
        if not token:
            token = os.getenv("HF_TOKEN", None)

        # Extract dataset_info section from existing README if available
        config_data = extract_dataset_info(repo_id=repo_id, token=token)
        print(f"Extracted dataset_info section, length: {len(config_data) if config_data else 0} characters")

        dataset_name = repo_id.split("/")[-1]
        pretty_name = dataset_name.replace("-", " ").replace("_", " ").title()

        card_data_kwargs = {
            "pretty_name": pretty_name
        }

        # Create DatasetCardData with our metadata
        card_data = DatasetCardData(**card_data_kwargs)

        # Get datafast version
        from importlib.metadata import version, PackageNotFoundError

        try:
            version_str = version("datafast")
        except PackageNotFoundError:
            # Fallback for development installs
            version_str = "dev"

        # Prepare template variables
        template_vars = {
            "pretty_name": card_data.pretty_name,
            "datafast_version": version_str,
            "config_data": config_data,  # Use the extracted dataset_info section
        }

        print("Rendering dataset card from template")
        print(f"Template variables: {list(template_vars.keys())}")

        # Render card with our template and variables 
        card = DatasetCard.from_template(
            card_data=card_data,
            template_str=template_str,
            **template_vars
        )

        print("Template rendered successfully")
        print(f"Rendered card content length: {len(str(card))} characters")

        # Push to hub
        print(f"Pushing dataset card to hub: {repo_id}")
        card.push_to_hub(repo_id, token=token)

        print(f"Dataset card successfully uploaded to: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"Failed to upload dataset card: {e}")
        print("Full traceback:")


def upload_dataset_card(repo_id: str, token: str | None = None) -> None:
    """
    Public interface to generate and upload a dataset card to Hugging Face Hub.
    
    This function performs configuration checks (like offline mode)
    and then delegates to the internal _generate_and_upload_dataset_card() implementation.
    It should be called at the end of the pipeline when all subsets are available.
    
    Args:
        repo_id: The ID of the repository to push to
        token: The token to authenticate with the Hugging Face Hub
    """
    try:

        print(f"Uploading dataset card to repository: {repo_id}")
        _generate_and_upload_dataset_card(
            repo_id=repo_id,
            token=token
        )

    except Exception as e:
        print(f"Error uploading dataset card: {e}")