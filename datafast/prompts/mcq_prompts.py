DEFAULT_TEMPLATES = [
    """You are an expert at creating exam questions. Your task is to come up with {num_samples} \
difficult multiple choice questions written in {language_name} in relation to the following document along with the correct answer.
The question should be self-contained, short and answerable. 
It is very important to have unique questions. No questions should be like 'what is X and what about Y?' or 'what is X and when did Y happen?'.
The answer must be short.
It must relate to the details of the document, but the question should contain enough context to be answerable for a person without the document.

### Document
{document}

Now come up with {num_samples} questions in relation to the document.
Make sure the questions are difficult, but answerable with a short answer.
Provide the correct answer for each question."""
]

DISTRACTOR_TEMPLATE = """
You are an expert in creating plausible but incorrect answers for multiple choice questions.
        
For the following question, and correct answer, generate 3 short, plausible but incorrect answers in {language_name}.
The incorrect answers should be wrong but not obviously wrong - they should be tempting distractors.
Do not provide explanations, just list the three incorrect answers one per line (without prefixing with A, B, C, or numbers etc.)

Question: {question}

Correct Answer: {correct_answer}

Now provide exactly 3 short, plausible but incorrect answers:
"""
