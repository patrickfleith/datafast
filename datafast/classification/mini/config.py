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
    "model_id": "claude-3-5-sonnet-latest"
}, {
    "provider": "openai",
    "model_id": "gpt-4o-mini"
}]


PROMPT_TEMPLATE = """I need text examples in order to train a machine learning model to classify between the following classes {labels_listing}. \
Your task is to generate {num_samples} texts which are diverse and representative of what could be encountered \
for the '{label_name}' class . {label_description}. \
Do not exagerate, and ensure that it a realistic text while it belongs to the described class."""


