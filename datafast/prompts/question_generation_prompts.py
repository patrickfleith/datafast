DOMAIN_TOPIC_SUBTOPIC_N_QUESTION_GENERATION_DEFAULT_TEMPLATES = [
    "You are an expert in {domain}. Generate a series of {num_samples} questions in {language_name} about {subtopic} in the context of {topic}",
    # "You are a distinguished professor of {topic} in the Departement of {domain}. Your task is to generate {num_samples} exam questions about the topic of {subtopic} in {language_name}.",
    # "Your task is to produce {num_samples} questions about {subtopic}, a subtopic under {topic} in the {domain} field. These questions must range from beginner to advanced levels, including short, medium, and longer forms. The questions are in {language_name}. Advanced queries should be highly detailed and use occasional technical jargon. Do not reuse the words “{subtopic}” or “{topic}” directly. Format the output as JSON with exactly {num_samples} entries, and provide no introduction, conclusion, or additional text—only the questions."
]

PERSONA_QUESTION_REFORMULATION_DEFAULT_TEMPLATE = "Your task is to reformulate the following question so that it is \
     plausible for the specified persona. If it already fits, leave it unchanged; otherwise, make some necessary edits \
     to really fit the persona (without overexagerating it). The question meaning should not be changed, and should \
     still pertains to the topic of {subtopic}. Ensure the reformulated question could be asked from the \
     viewpoint of {persona}. Respond only with the final question—no additional text. Here is the question: {question}"

SIMULATED_ASSISTANT_DEFAULT_TEMPLATE = """You are specialized in the domain of {domain} and in particular about {topic} and specifically about {subtopic}
You task to answer to inquiries that showcase your depth of knowledge and ability to communicate complex information very concisely, clearly and effectively.
Provide clear, very concise answers that directly address the question.
If is helps, and only if you are sure about it, you can include relevant facts, numbers, or examples where appropriate to enhance understanding.
Here is the question to answer: {question}
"""

USER_FOLLOWUP_PROMPT_TEMPLATE = """Act as if you are very skilled in role playing what a human user would be asking as a followup questions to an AI to continue the conversation. 
You are role playing this persona: {persona}
Here is a summary of a conversation between a user and an intelligent assistant:
{dialog_summary}
Above is a conversation summary between a user and an intelligent assistant about the topic of {subtopic} in the {domain} domain.
Now suppose you are the human user, say something to continue the conversation based on given context.
Make the follow-up question short and realistic given you role. Your query should feel 
natural and related to the previous conversation. You can ask for more details, clarification, or further explanation, another example, digging deeper etc. etc."""