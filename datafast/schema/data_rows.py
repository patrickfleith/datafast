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


LabelType = Union[str, list[str], list[int]]


class TextRow(BaseModel):
    """Row for storing generated text data."""
    text: str
    text_source: TextSource = TextSource.SYNTHETIC
    model_id: Optional[str] = None
    uuid: UUID = Field(default_factory=uuid4)
    metadata: dict[str, str] = Field(default_factory=dict)


class TextClassificationRow(BaseModel):
    text: str
    label: LabelType  # Must be either str, list[str], or list[int]
    model_id: Optional[str] = None
    label_source: LabelSource = LabelSource.SYNTHETIC
    confidence_scores: Optional[dict[str, float]] = Field(default_factory=dict)

    # System and metadata fields
    uuid: UUID = Field(default_factory=uuid4)
    metadata: dict[str, str] = Field(default_factory=dict)
