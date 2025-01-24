PROMPT_TEMPLATE = """Imagine {num_entries} differents and realistic sentences which could have been written by the following person:
Persona: {persona}
Writting style: {style}
Context: {context}
Medium: {medium}
Intent: {intent}

Instructions:
Each text entry must be short (1 to 2) sentences max.
The core idea in each text entry must be different from the others.
The language must be {language}
"""