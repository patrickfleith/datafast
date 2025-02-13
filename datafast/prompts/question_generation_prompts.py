DOMAIN_TOPIC_SUBTOPIC_N_QUESTION_GENERATION_DEFAULT_TEMPLATES = [
    "You are an expert in {domain}. Generate a series of {n_questions} questions about {subtopic} in the context of {topic}",
    "You are a distinguished professor of {topic} in the Departement of {domain}. Your task is to generate {n_questions} exam questions about the topic of {subtopic}.",
    "Your task is to produce {n_questions} questions about {subtopic}, a subtopic under {topic} in the {domain} field. These questions must range from beginner to advanced levels, including short, medium, and longer forms. Advanced queries should be highly detailed and use occasional technical jargon. Do not reuse the words “{subtopic}” or “{topic}” directly. Format the output as JSON with exactly {n_questions} entries, and provide no introduction, conclusion, or additional text—only the questions."
]

PERSONA_QUESTION_REFORMULATION_DEFAULT_TEMPLATE = "Your task is to confirm the user’s question is plausible for the specified persona. \
    If it already fits, leave it unchanged; otherwise, make some necessary edits to really fit the persona (without overexagerating it). \
        The question pertains to {subtopic} and is asked from the viewpoint of {persona}. Respond only with the final question—no additional text."

SIMULATED_ASSISTANT_DEFAULT_TEMPLATE = """You are a helpful AI assistant specialized in the domain of {domain} and in particular about {topic} and specifically about {subtopic}
You task to answer to inquiries that showcase your depth of knowledge and ability to communicate complex information very concisely, clearly and effectively.

Instructions:

Provide clear, very concise answers that directly address the users' questions.
If is helps, and only if you are sure about it, you can include relevant facts, numbers, or examples where appropriate to enhance understanding."""


USER_SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant, in particular you are very skilled in role playing what a human user would be asking as a followup questions to an AI to continue the conversation. You are role playing this persona: {persona}"""

USER_HUMAN_PROMPT_TEMPLATE = """Here is a summary of a conversation between a user and an intelligent assistant.

{dialog_summary}

Above is a conversation summary between a user and an intelligent assistant about the topic of {subtopic} in the {domain} domain.
Now suppose you are the human user, say something to continue the conversation based on given context.
Make the follow-up question short and realistic given that you should act as {persona}. Your query should feel 
natural and related to the previous conversation. You can ask for more details, clarification, or further explanation, another example, digging deeper etc. etc."""