"""
nlp.py
------
Morphological analysis using spaCy (en_core_web_sm)
plus simple rule-based word generation.
"""

import spacy

# Load the English model once at import time
nlp = spacy.load("en_core_web_sm")


def generate_word_form(lemma: str) -> str:
    """
    Apply simple plural-generation rules:
      - ends with 'y' -> replace 'y' with 'ies'
      - ends with 's' -> add 'es'
      - otherwise     -> add 's'
    """
    if not lemma:
        return lemma

    if lemma.endswith("y"):
        return lemma[:-1] + "ies"
    elif lemma.endswith("s"):
        return lemma + "es"
    else:
        return lemma + "s"


def analyze_text(text: str) -> dict:
    """
    Process text with spaCy and return:
      - original_text
      - tokens: list of {word, lemma, pos, morphology}
      - generated_words: list of {lemma, generated}
    """
    doc = nlp(text)

    tokens = []
    generated_words = []

    for token in doc:
        # Morphological analysis for EVERY token
        tokens.append({
            "word": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "morphology": str(token.morph) if token.morph else ""
        })

        # Generate new word forms only from alphabetic tokens
        if token.is_alpha:
            generated_words.append({
                "lemma": token.lemma_,
                "generated": generate_word_form(token.lemma_)
            })

    return {
        "original_text": text,
        "tokens": tokens,
        "generated_words": generated_words
    }