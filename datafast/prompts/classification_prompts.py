DEFAULT_TEMPLATES = [
    """I need text examples in order to train a machine learning model to classify \
between the following classes {labels_listing}. Your task is to generate {num_samples} \
texts written in {language_name} which are diverse and representative of what could be \
encountered for the '{label_name}' class. {label_description}. Do not exagerate, and \
ensure that it a realistic text while it belongs to the described class."""
]
