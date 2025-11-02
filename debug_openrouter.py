import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from typing import List

class MCQQuestion(BaseModel):
    """Multiple choice question with one correct and three incorrect answers."""
    question: str = Field(description="The question")
    correct_answer: str = Field(description="The correct answer")
    incorrect_answers: List[str] = Field(description="List of 3 incorrect answers")

    @field_validator('incorrect_answers')
    def check_incorrect_count(cls, v: List[str]) -> List[str]:
        """Validate that exactly 3 incorrect answers are provided."""
        if len(v) != 3:
            raise ValueError("Must provide exactly 3 incorrect answers")
        return v


class MCQSet(BaseModel):
    """Set of multiple choice questions."""
    questions: List[MCQQuestion] = Field(description="List of MCQ questions")

    @field_validator('questions')
    def check_questions_count(cls, v: List[MCQQuestion]) -> List[MCQQuestion]:
        if len(v) != 3:
            raise ValueError("Must provide exactly 3 questions")
        return v


def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        return s.strip("`")
    return s


prompt = (
    "Generate exactly 3 multiple choice questions about machine learning.\n\n"
    "For each question, provide: the question, one correct answer, and three plausible but incorrect answers.\n\n"
)

response = litellm.completion(
    model="openrouter/z-ai/glm-4.6",
    messages=[{"role": "user", "content": prompt}],
    response_format=MCQSet
)

content = response.choices[0].message.content
content = _strip_code_fences(content)
parsed = MCQSet.model_validate_json(content)
print(parsed.model_dump_json(indent=2))
