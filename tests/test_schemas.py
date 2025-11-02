"""Shared Pydantic schemas for LLM provider tests."""
from typing import List
from pydantic import BaseModel, Field, field_validator


class SimpleResponse(BaseModel):
    """Simple response model for testing structured output."""
    answer: str = Field(description="The answer to the question")
    reasoning: str = Field(description="The reasoning behind the answer")


class Attribute(BaseModel):
    """Attribute of a landmark with value and importance."""
    name: str = Field(description="Name of the attribute")
    value: str = Field(description="Value of the attribute")
    importance: float = Field(description="Importance score between 0 and 1")

    @field_validator('importance')
    @classmethod
    def check_importance(cls, v: float) -> float:
        """Validate importance is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Importance must be between 0 and 1")
        return v


class LandmarkInfo(BaseModel):
    """Information about a landmark with attributes."""
    name: str = Field(description="The name of the landmark")
    location: str = Field(description="Where the landmark is located")
    description: str = Field(description="A brief description of the landmark")
    year_built: int | None = Field(
        None, description="Year when the landmark was built")
    attributes: List[Attribute] = Field(
        description="List of attributes about the landmark")
    visitor_rating: float = Field(
        description="Average visitor rating from 0 to 5")

    @field_validator('visitor_rating')
    @classmethod
    def check_rating(cls, v: float) -> float:
        """Validate rating is between 0 and 5."""
        if not 0 <= v <= 5:
            raise ValueError("Rating must be between 0 and 5")
        return v


class PersonaContent(BaseModel):
    """Generated content for a persona including tweets and bio."""
    tweets: List[str] = Field(description="List of 5 tweets for the persona")
    bio: str = Field(description="Biography for the persona")

    @field_validator('tweets')
    @classmethod
    def check_tweets_count(cls, v: List[str]) -> List[str]:
        """Validate that exactly 5 tweets are provided."""
        if len(v) != 5:
            raise ValueError("Must provide exactly 5 tweets")
        return v


class QAItem(BaseModel):
    """Question and answer pair."""
    question: str = Field(description="The question")
    answer: str = Field(description="The correct answer")


class QASet(BaseModel):
    """Set of questions and answers."""
    questions: List[QAItem] = Field(description="List of question-answer pairs")

    @field_validator('questions')
    @classmethod
    def check_qa_count(cls, v: List[QAItem]) -> List[QAItem]:
        """Validate that exactly 5 Q&A pairs are provided."""
        if len(v) != 5:
            raise ValueError("Must provide exactly 5 question-answer pairs")
        return v


class MCQQuestion(BaseModel):
    """Multiple choice question with one correct and three incorrect answers."""
    question: str = Field(description="The question")
    correct_answer: str = Field(description="The correct answer")
    incorrect_answers: List[str] = Field(description="List of 3 incorrect answers")

    @field_validator('incorrect_answers')
    @classmethod
    def check_incorrect_count(cls, v: List[str]) -> List[str]:
        """Validate that exactly 3 incorrect answers are provided."""
        if len(v) != 3:
            raise ValueError("Must provide exactly 3 incorrect answers")
        return v


class MCQSet(BaseModel):
    """Set of multiple choice questions."""
    questions: List[MCQQuestion] = Field(description="List of MCQ questions")

    @field_validator('questions')
    @classmethod
    def check_questions_count(cls, v: List[MCQQuestion]) -> List[MCQQuestion]:
        """Validate that exactly 3 questions are provided."""
        if len(v) != 3:
            raise ValueError("Must provide exactly 3 questions")
        return v
