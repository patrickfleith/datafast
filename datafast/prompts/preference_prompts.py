"""
Prompt templates for preference dataset generation.
"""

# Default prompt templates for generating questions from documents
QUESTION_GENERATION_TEMPLATES = [
    "Based on the following document, generate {num_samples} clear and specific questions in {language_name} that would require detailed responses. The questions should be diverse and cover different aspects of the document content:\n\n{document}",
    "You are an expert interviewer. Given this document, create {num_samples} thoughtful questions in {language_name} that would elicit detailed and informative responses. Focus on different aspects of the content:\n\n{document}",
    "Generate {num_samples} questions in {language_name} based on this document that would be suitable for testing an AI assistant's ability to provide helpful, accurate, and comprehensive answers:\n\n{document}"
]

# Prompt template for generating high-quality (chosen) responses
CHOSEN_RESPONSE_TEMPLATE = """You are an expert AI assistant known for providing exceptionally helpful, accurate, and comprehensive responses.
Given the document and question below, provide a concise and well-structured answer in {language_name}:

DOCUMENT:
{document}

QUESTION:
{question}

Your response should be helpful, concise, well-organized, and directly address all aspects of the question."""

# Prompt template for generating lower-quality (rejected) responses
REJECTED_RESPONSE_TEMPLATE = """Provide a response in {language_name} to the following question based on the document:

DOCUMENT:
{document}

QUESTION:
{question}"""

# Prompt template for evolutionary instruction refinement
EVOLUTION_PROMPT = """Your task is to evolve both the question and answer to create a more challenging and interesting version.

ORIGINAL DOCUMENT:
{document}

ORIGINAL QUESTION:
{question}

ORIGINAL ANSWER:
{answer}

First, improve the question to make it more specific, nuanced, or complex while still being answerable from the document.
Then, provide an improved answer to your evolved question that is more comprehensive, accurate, and helpful than the original.

Your response should include both the improved question and improved answer."""

# Prompt template for LLM judge scoring
JUDGE_PROMPT = """You are an expert evaluator assessing the quality of responses from an AI assistant to user queries.
Rate the following response on a scale from 1 to 10, where 1 is extremely poor and 10 is excellent.

DOCUMENT:
{document}

QUESTION:
{question}

RESPONSE TO EVALUATE:
{response}

Consider these criteria in your evaluation:
- Accuracy: Does the response provide correct information based on the document?
- Completeness: Does the response address all aspects of the question?
- Clarity: Is the response well-organized and easy to understand?
- Conciseness: Is the response concise and to the point?
- Helpfulness: Would the response be genuinely useful to someone asking this question?

Provide a brief assessment of the response, highlighting specific strengths and weaknesses.

YOUR SCORE MUST BE AN INTEGER BETWEEN 1 AND 10 INCLUSIVE. Do not provide decimal or fractional scores.

Format your response with your assessment followed by the score on its own line."""

