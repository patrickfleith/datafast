DEFAULT_TEMPLATES = ["""You are an expert at creating exam questions. Your task is to come up with {num_samples} \
difficult multiple choice questions written in {language_name} in relation to the following document along with the correct answer.
The questions should be self-contained, short and answerable. 
It is very important to have unique questions. No questions should be like 'what is X and what about Y?' or 'what is X and when did Y happen?'.
The correct answers must be short.
It must relate to the details of the document, but the question should contain enough context to be answerable for a person without the document.

### Document
{document}

Now come up with {num_samples} questions in relation to the document.
Make sure the questions are difficult, but answerable with a short answer.
Provide the correct answer for each question.
Do not list candidate answers as part of the questions.
Do not include numbers or letters (A, B, C, D, etc.) as prefix in the questions or answers.
"""]
    
# Template for more contextualized questions
CONTEXTUALISED_TEMPLATES = ["""You are an expert at creating exam questions. Your task is to come up with {num_samples} \
 multiple choice questions written in {language_name} in relation to the following document along with the correct answers.
The questions should be self-contained, short and answerable. 
It is very important to have unique questions. No questions should be like 'what is X and what about Y?' or 'what is X and when did Y happen?'.
The answers must be short.
It must relate to the details of the document. However questions should never contain wording as reference to the document like "according to the report, 'in this paper', 'in the document', etc.
Make sure to write the questions to include some very brief context, like if the person asking the questions would be explaining the context in which the question arise very concisely. This is just to remove ambiguity like if the question was provided in an exam.

### Context
{context}

### Document
{document}

Now come up with {num_samples} contextualized questions in relation to the document.
Make sure the questions are difficult, but answerable with a short answer.
Provide the correct answer for each question.
Do not list candidate answers as part of the questions.
Do not include numbers or letters (A, B, C, D, etc.) as prefix in the questions or answers.
"""
]

DISTRACTOR_TEMPLATE = """
You are an expert in creating plausible but incorrect answers for multiple choice questions.
        
For the following question, and correct answer, generate 3 short, plausible but incorrect answers in {language_name}.
The incorrect answers should be wrong but not obviously wrong - they should be tempting distractors.
Do not provide explanations, just list the three incorrect answers one per line (without prefixing with A, B, C, or numbers etc.)

Question: {question}

Correct Answer: {correct_answer}

Now provide exactly 3 short, plausible but incorrect answers:
Make sure the the length, structure, or writing style of the incorrect answers are not all at the same time obviously different from the correct answer (e.g. if the correct answer does not contain a trailing period, also avoid trailing periods in the incorrect answers, or make sure their length are similar, the way numbers or names are written, etc.)
Do not include numbers or letters (A, B, C, D, etc.) as prefix in the incorrect answers.
"""
