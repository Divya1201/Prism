def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing:
    - lowercase
    - remove extra spaces
    """

    if not text:
        return ""

    #text = text.lower()
    text = " ".join(text.split())

    return text
