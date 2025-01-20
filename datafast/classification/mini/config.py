LABELS = [
    {
        'label_id': 0,
        'name': 'concise',
        'description': "Concise text is characterized by clarity, precision, and brevity. It communicates ideas directly, in a very compact manner using only the words necessary to convey the intended message."
    },
    {
        'label_id': 1,
        'name': 'vebose',
        'description': "Verbose in text is often includes some redundancy, filler words, excessive qualifiers, unnecessary adjectives or adverbs, irrelevant information, and repetition of known context."
    }
]

LANGUAGES = {
    "fr": "French",
    "es": "Spanish",
    "en": "English"
}

MODEL_IDS = [{
    "provider": "google",
    "model_id": "gemini-1.5-flash"
}, {
    "provider": "anthropic",
    "model_id": "claude-3.5-sonnet"
}, {
    "provider": "openai",
    "model_id": "gpt-4o-mini"
}]


PROMPT_TEMPLATE = """I need text examples in order to train a machine learning model to classify between the following classes {labels_listing}. \
Your task is to generate {num_samples} texts which are diverse and representative of what could be encountered \
for the '{label_name}' class . {label_description}. \
Do not exagerate, and ensure that it a realistic text while it belongs to the described class."""



CONCISE_GENERATED_PROMPT_TEMPLATE = """I am creating a dataset to train a classifier that distinguishes concise texts from verbose texts. Your task is to generate concise, clear, and professional texts. Each example should communicate the essential information directly, avoiding unnecessary elaboration or redundant details. The tone should remain professional and precise.

Generate {number} examples of concise texts. Each example should:

Be a realistic, coherent response to an imagined query, scenario, or description.
Focus only on the key points and avoid adding extra context, explanations, or non-essential details.
Use short, clear sentence structures to convey the message efficiently.
Although short, don't cut the meet. Conciseness mean precise and specific in our case.

For example:

A brief explanation of a process or concept.
A high-level summary of an event or scenario.
A concise email or message stating only the critical information.
Avoid repeating ideas across examples and ensure each one is unique and succinct."""



