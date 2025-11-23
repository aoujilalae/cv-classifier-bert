# src/cv_match_bert.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

# Charger le modèle (léger et performant)
_model = None

def get_model(model_name: str = "all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

def encode_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
    """
    Encode une liste de textes en vecteurs avec Sentence-BERT.
    Retourne un array numpy de formes (n_texts, dim).
    """
    model = get_model(model_name)
    # encode renvoie numpy array si show_progress_bar=False (par défaut)
    vecs = model.encode(texts, show_progress_bar=False)
    return np.array(vecs)

def score_pair(cv_text: str, offre_text: str, model_name: str = "all-MiniLM-L6-v2"):
    """
    Score entre un CV et une offre (single pair).
    """
    cv_vec = encode_texts([cv_text], model_name=model_name)
    offre_vec = encode_texts([offre_text], model_name=model_name)
    score = cosine_similarity(cv_vec, offre_vec)[0][0]
    return float(score)

def score_batch(cv_texts: List[str], offre_text: str, model_name: str = "all-MiniLM-L6-v2"):
    """
    Score un batch de CVs contre une seule offre.
    Retourne une liste de scores alignée sur cv_texts.
    """
    if len(cv_texts) == 0:
        return []
    cv_vecs = encode_texts(cv_texts, model_name=model_name)  # (n, d)
    offre_vec = encode_texts([offre_text], model_name=model_name)  # (1, d)
    # cosine_similarity peut calculer n x 1 matrix
    scores = cosine_similarity(cv_vecs, offre_vec).reshape(-1)
    return [float(s) for s in scores]

if __name__ == "__main__":
    # Petit test local
    cvs = [
        "Compétences : Python, Machine Learning, Git, Pandas\nExpérience : 1 an en data analysis",
        "Compétences : Photoshop, Illustrator, UI/UX design\nExpérience : 3 ans en design"
    ]
    offre = "Nous recherchons un développeur Python avec expérience en Pandas, Git et machine learning."
    print("Encodage et scoring demo...")
    scores = score_batch(cvs, offre)
    for i, s in enumerate(scores):
        print(f"CV {i+1} -> score {s:.3f}")
