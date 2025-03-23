from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import Union, Optional
from enum import Enum


class LabelSource(str, Enum):
    SYNTHETIC = "synthetic"
    VERIFIED = "verified"
    HUMAN = "human"
    CONSENSUS = "consensus"


class TextSource(str, Enum):
    SYNTHETIC = "synthetic"
    VERIFIED = "verified"
    HUMAN = "human"
    CONSENSUS = "consensus"


class MCQSource(str, Enum):
    SYNTHETIC = "synthetic"
    VERIFIED = "verified"
    HUMAN = "human"
    CONSENSUS = "consensus"


LabelType = Union[str, list[str], list[int]]


class TextRow(BaseModel):
    """Row for storing generated text data."""

    text: str
    text_source: TextSource = TextSource.SYNTHETIC
    model_id: Optional[str] = None
    uuid: UUID = Field(default_factory=uuid4)
    metadata: dict[str, str] = Field(default_factory=dict)


class ChatRow(BaseModel):
    """Row for storing generated conversations"""

    opening_question: str
    messages: list[dict[str, str]]
    model_id: Optional[str] = None
    uuid: UUID = Field(default_factory=uuid4)
    metadata: dict[str, str] = Field(default_factory=dict)
    persona: str


class TextClassificationRow(BaseModel):
    text: str
    label: LabelType  # Must be either str, list[str], or list[int]
    model_id: Optional[str] = None
    label_source: LabelSource = LabelSource.SYNTHETIC
    confidence_scores: Optional[dict[str, float]] = Field(default_factory=dict)

    # System and metadata fields
    uuid: UUID = Field(default_factory=uuid4)
    metadata: dict[str, str] = Field(default_factory=dict)


class MCQRow(BaseModel):
    source_document: str
    question: str
    correct_answer: str
    incorrect_answer_1: str
    incorrect_answer_2: str
    incorrect_answer_3: str
    model_id: Optional[str] = None
    mcq_source: MCQSource = MCQSource.SYNTHETIC
    uuid: UUID = Field(default_factory=uuid4)
    metadata: dict[str, str] = Field(default_factory=dict)


class PreferenceSource(str, Enum):
    SYNTHETIC = "synthetic"
    VERIFIED = "verified"
    HUMAN = "human"
    CONSENSUS = "consensus"


class PreferenceRow(BaseModel):
    """Row for storing preference data with chosen and rejected responses."""
    
    input_document: str
    question: str
    chosen_response: str
    rejected_response: str
    preference_source: PreferenceSource = PreferenceSource.SYNTHETIC
    chosen_model_id: Optional[str] = None
    rejected_model_id: Optional[str] = None
    
    # Optional judge-related fields
    chosen_response_score: Optional[int] = None
    rejected_response_score: Optional[int] = None
    chosen_response_assessment: Optional[str] = None
    rejected_response_assessment: Optional[str] = None

    uuid: UUID = Field(default_factory=uuid4)
    metadata: dict[str, str] = Field(default_factory=dict)