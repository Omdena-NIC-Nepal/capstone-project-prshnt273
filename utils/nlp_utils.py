import spacy
from spacy import displacy
from collections import Counter
import pandas as pd

# Load English language model
nlp = spacy.load("en_core_web_sm")

def analyze_text(text):
    """Analyze climate-related text using spaCy"""
    doc = nlp(text)
    
    # Extract entities, nouns, verbs
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    verbs = [token.text for token in doc if token.pos_ == "VERB"]
    
    # Get most common terms
    words = [token.text for token in doc if token.is_alpha]
    word_freq = Counter(words).most_common(10)
    
    return {
        "entities": entities,
        "nouns": nouns,
        "verbs": verbs,
        "word_freq": word_freq,
        "doc": doc
    }