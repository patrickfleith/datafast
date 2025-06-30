"""
Specialized Gradio inspectors for each dataset type in datafast.

Usage:
    from datafast.inspectors import inspect_classification_dataset, inspect_mcq_dataset, ...
    inspect_classification_dataset(dataset)
    
    # Or use random ordering:
    inspect_classification_dataset(dataset, random=True)

Each function launches a Gradio app tailored to the row structure of the dataset.
"""
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Tuple, Type, TypeVar, Union, cast
import re
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datasets import (
        DatasetBase,
        ClassificationDataset, 
        MCQDataset, 
        PreferenceDataset, 
        RawDataset,
        UltrachatDataset
    )
    import gradio as gr

# Type variables for generic typing
T = TypeVar('T')
DatasetT = TypeVar('DatasetT', bound='DatasetBase')


class BaseInspector(Generic[DatasetT]):
    """Base class for all dataset inspectors."""
    
    # Class variables to be overridden by subclasses
    title: str = "Dataset Inspector"
    
    def __init__(self, dataset: DatasetT, random: bool = False):
        """
        Initialize the inspector.
        
        Args:
            dataset: The dataset to inspect
            random: If True, examples will be shown in random order
        """
        try:
            import gradio as gr
            self.gr = gr
        except ImportError as e:
            raise ImportError("Gradio is required for inspection. Install with 'pip install gradio'.") from e
            
        if not hasattr(dataset, 'data_rows') or not dataset.data_rows:
            raise ValueError("No data rows to inspect. Generate or load data first.")
            
        self.dataset = dataset
        self.random = random
        
        # Convert data rows to dicts for display
        self.examples = [
            row.model_dump() if hasattr(row, 'model_dump') 
            else row.dict() if hasattr(row, 'dict') 
            else row for row in dataset.data_rows
        ]
        self.total = len(self.examples)
        
        # Set up display order (sequential or random)
        if random and self.total > 1:
            # Create a permutation of indices
            random_indices = np.random.permutation(self.total)
            self.display_order = list(random_indices)
            self.ordering_label = "(Random Order)"
        else:
            # Sequential order
            self.display_order = list(range(self.total))
            self.ordering_label = ""
    
    def get_example(self, idx: int) -> Dict[str, Any]:
        """Get the example at the specified display index."""
        idx = max(0, min(idx, self.total - 1))
        # Map display index to actual example index
        example_idx = self.display_order[idx]
        return self.examples[example_idx]
    
    def get_index_label(self, idx: int) -> str:
        """Get the index label for the UI."""
        idx = max(0, min(idx, self.total - 1))
        return f"Example {idx+1} / {self.total} {self.ordering_label}"
        
    def show_example(self, idx: int) -> Tuple:
        """
        Get the data to display for a specific example.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement show_example")
    
    def create_ui_components(self) -> List:
        """
        Create the UI components for the inspector.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement create_ui_components")
    
    def launch(self) -> None:
        """Launch the Gradio inspector app."""
        gr = self.gr
        
        with gr.Blocks() as demo:
            idx_state = gr.State(0)
            gr.Markdown(f"# {self.title}")
            
            # Create label for current example
            idx_label = gr.Markdown()
            
            # Create UI components defined by the subclass
            components = self.create_ui_components()
            
            # Add navigation buttons
            with gr.Row():
                prev_btn = gr.Button("Previous")
                next_btn = gr.Button("Next")
            
            # Define update function that gets data and updates UI
            def update(idx):
                return self.show_example(idx) + (idx,)
            
            # Connect navigation buttons
            prev_btn.click(
                lambda idx: max(0, idx-1), idx_state, idx_state
            ).then(
                update, idx_state, [idx_label] + components + [idx_state]
            )
            
            next_btn.click(
                lambda idx: min(self.total-1, idx+1), idx_state, idx_state
            ).then(
                update, idx_state, [idx_label] + components + [idx_state]
            )
            
            # Initial display
            demo.load(update, idx_state, [idx_label] + components + [idx_state])
        
        demo.launch()

class ClassificationInspector(BaseInspector['ClassificationDataset']):
    """Inspector for ClassificationDataset showing text, label, model_id, and metadata."""
    
    title = "Classification Dataset Inspector"
    
    def create_ui_components(self) -> List:
        """Create UI components specific to classification data."""
        gr = self.gr
        text = gr.Textbox(label="Text", interactive=False)
        label = gr.Textbox(label="Label", interactive=False)
        model_id = gr.Textbox(label="Model ID", interactive=False)
        metadata = gr.JSON(label="Metadata")
        return [text, label, model_id, metadata]
    
    def show_example(self, idx: int) -> Tuple[str, str, str, str, Dict]:
        """Extract data from the example to display."""
        row = self.get_example(idx)
        return (
            self.get_index_label(idx),
            row.get("text", ""),
            str(row.get("label", "")),
            row.get("model_id", ""),
            row.get("metadata", {})
        )


def inspect_classification_dataset(dataset: 'ClassificationDataset', random: bool = False) -> None:
    """
    Launch a Gradio inspector for a ClassificationDataset object.
    Shows text, label, model_id, and metadata fields.
    
    Args:
        dataset: The ClassificationDataset to inspect
        random: If True, examples will be shown in random order instead of sequential order.
               Default is False (sequential order).
    """
    inspector = ClassificationInspector(dataset, random)
    inspector.launch()

class MCQInspector(BaseInspector['MCQDataset']):
    """Inspector for MCQDataset showing question, correct/incorrect answers, model_id, and metadata."""
    
    title = "MCQ Dataset Inspector"
    
    def create_ui_components(self) -> List:
        """Create UI components specific to MCQ data."""
        gr = self.gr
        question = gr.Textbox(label="Question", interactive=False)
        correct = gr.Textbox(label="Correct Answer", interactive=False)
        inc1 = gr.Textbox(label="Incorrect Answer 1", interactive=False)
        inc2 = gr.Textbox(label="Incorrect Answer 2", interactive=False)
        inc3 = gr.Textbox(label="Incorrect Answer 3", interactive=False)
        model_id = gr.Textbox(label="Model ID", interactive=False)
        metadata = gr.JSON(label="Metadata")
        return [question, correct, inc1, inc2, inc3, model_id, metadata]
    
    def show_example(self, idx: int) -> Tuple[str, str, str, str, str, str, str, Dict]:
        """Extract data from the example to display."""
        row = self.get_example(idx)
        return (
            self.get_index_label(idx),
            row.get("question", ""),
            row.get("correct_answer", ""),
            row.get("incorrect_answer_1", ""),
            row.get("incorrect_answer_2", ""),
            row.get("incorrect_answer_3", ""),
            row.get("model_id", ""),
            row.get("metadata", {})
        )


def inspect_mcq_dataset(dataset: 'MCQDataset', random: bool = False) -> None:
    """
    Launch a Gradio inspector for an MCQDataset object.
    Shows question, correct/incorrect answers, model_id, and metadata.
    
    Args:
        dataset: The MCQDataset to inspect
        random: If True, examples will be shown in random order instead of sequential order.
               Default is False (sequential order).
    """
    inspector = MCQInspector(dataset, random)
    inspector.launch()

class PreferenceInspector(BaseInspector['PreferenceDataset']):
    """Inspector for PreferenceDataset showing input, questions, chosen/rejected responses, scores, etc."""
    
    title = "Preference Dataset Inspector"
    
    def create_ui_components(self) -> List:
        """Create UI components specific to preference data."""
        gr = self.gr
        input_doc = gr.Textbox(label="Input Document", interactive=False)
        question = gr.Textbox(label="Question", interactive=False)
        chosen = gr.Textbox(label="Chosen Response", interactive=False)
        rejected = gr.Textbox(label="Rejected Response", interactive=False)
        chosen_model = gr.Textbox(label="Chosen Model ID", interactive=False)
        rejected_model = gr.Textbox(label="Rejected Model ID", interactive=False)
        chosen_score = gr.Number(label="Chosen Score", interactive=False)
        rejected_score = gr.Number(label="Rejected Score", interactive=False)
        chosen_assess = gr.Textbox(label="Chosen Assessment", interactive=False)
        rejected_assess = gr.Textbox(label="Rejected Assessment", interactive=False)
        metadata = gr.JSON(label="Metadata")
        return [
            input_doc, question, chosen, rejected, 
            chosen_model, rejected_model, chosen_score, rejected_score,
            chosen_assess, rejected_assess, metadata
        ]
    
    def show_example(self, idx: int) -> Tuple[str, str, str, str, str, str, str, int, int, str, str, Dict]:
        """Extract data from the example to display."""
        row = self.get_example(idx)
        return (
            self.get_index_label(idx),
            row.get("input_document", ""),
            row.get("question", ""),
            row.get("chosen_response", ""),
            row.get("rejected_response", ""),
            row.get("chosen_model_id", ""),
            row.get("rejected_model_id", ""),
            row.get("chosen_response_score", 0),
            row.get("rejected_response_score", 0),
            row.get("chosen_response_assessment", ""),
            row.get("rejected_response_assessment", ""),
            row.get("metadata", {})
        )


def inspect_preference_dataset(dataset: 'PreferenceDataset', random: bool = False) -> None:
    """
    Launch a Gradio inspector for a PreferenceDataset object.
    Shows input_document, question, chosen/rejected responses, model_ids, scores, and metadata.
    
    Args:
        dataset: The PreferenceDataset to inspect
        random: If True, examples will be shown in random order instead of sequential order.
               Default is False (sequential order).
    """
    inspector = PreferenceInspector(dataset, random)
    inspector.launch()

class RawInspector(BaseInspector['RawDataset']):
    """Inspector for RawDataset showing text, text_source, and metadata."""
    
    title = "Raw Dataset Inspector"
    
    def create_ui_components(self) -> List:
        """Create UI components specific to raw text data."""
        gr = self.gr
        text = gr.Textbox(label="Text", interactive=False)
        text_source = gr.Textbox(label="Text Source", interactive=False)
        metadata = gr.JSON(label="Metadata")
        return [text, text_source, metadata]
    
    def show_example(self, idx: int) -> Tuple[str, str, str, Dict]:
        """Extract data from the example to display."""
        row = self.get_example(idx)
        return (
            self.get_index_label(idx),
            row.get("text", ""),
            row.get("text_source", ""),
            row.get("metadata", {})
        )


def inspect_raw_dataset(dataset: 'RawDataset', random: bool = False) -> None:
    """
    Launch a Gradio inspector for a RawDataset object.
    Shows text and metadata fields.
    
    Args:
        dataset: The RawDataset to inspect
        random: If True, examples will be shown in random order instead of sequential order.
               Default is False (sequential order).
    """
    inspector = RawInspector(dataset, random)
    inspector.launch()

class UltrachatInspector(BaseInspector['UltrachatDataset']):
    """Inspector for UltrachatDataset showing chat conversation, model_id, and metadata."""
    
    title = "Ultrachat Dataset Inspector"
    
    def create_ui_components(self) -> List:
        """Create UI components specific to Ultrachat data."""
        gr = self.gr
        conversation = gr.Textbox(label="Conversation", interactive=False, lines=15)
        model_id = gr.Textbox(label="Model ID", interactive=False)
        metadata = gr.JSON(label="Metadata")
        return [conversation, model_id, metadata]
    
    def format_conversation(self, conversation: List[Dict]) -> str:
        """Format the conversation messages for display."""
        result = ""
        for message in conversation:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            if role == "system":
                result += f"🔧 System: {content}\n\n"
            elif role == "user":
                result += f"👤 User: {content}\n\n"
            elif role == "assistant":
                result += f"🤖 Assistant: {content}\n\n"
            else:
                result += f"{role.capitalize()}: {content}\n\n"
        return result
    
    def show_example(self, idx: int) -> Tuple[str, str, str, Dict]:
        """Extract data from the example to display."""
        row = self.get_example(idx)
        conversation = row.get("conversation", [])
        formatted_convo = self.format_conversation(conversation)
        return (
            self.get_index_label(idx),
            formatted_convo,
            row.get("model_id", ""),
            row.get("metadata", {})
        )


def inspect_ultrachat_dataset(dataset: 'UltrachatDataset', random: bool = False) -> None:
    """
    Launch a Gradio inspector for an UltrachatDataset object.
    Shows the chat history and metadata.
    
    Args:
        dataset: The UltrachatDataset to inspect
        random: If True, examples will be shown in random order instead of sequential order.
               Default is False (sequential order).
    """
    inspector = UltrachatInspector(dataset, random)
    inspector.launch()

