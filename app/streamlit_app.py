# app/streamlit_app.py
import streamlit as st
import pandas as pd
import sys
import os

# Ajouter le dossier src au PATH pour que Python trouve cv_match_bert.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import io
import matplotlib.pyplot as plt

from cv_match_bert import score_batch, score_pair

st.set_page_config(page_title="Classifieur CV — Match / Pas Match", layout="wide")
st.title("Classifieur de CV — Match / Pas Match (Sentence-BERT)")

st.sidebar.header("Paramètres")
seuil = st.sidebar.slider("Seuil de similarité (≥ = MATCH)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
model_name = st.sidebar.selectbox("Modèle Sentence-BERT", ["all-MiniLM-L6-v2"], index=0)
st.sidebar.markdown("Upload plusieurs CVs (txt) et collez l'offre. Le système calcule un score (0–1).")

# Section principale
st.markdown("### 1) Charger les CVs")
uploaded_cv_files = st.file_uploader("Téléverse un ou plusieurs CV (.txt) — accept multiple", type=["txt"], accept_multiple_files=True)

st.markdown("### 2) Indiquer l'offre d'emploi")
offre_file = st.file_uploader("Ou téléverse un fichier d'offre (.txt)", type=["txt"], accept_multiple_files=False)
offre_text_input = st.text_area("Ou colle ici le texte de l'offre (priorité si rempli)")

# Préparer l'offre_text
offre_text = ""
if offre_text_input and len(offre_text_input.strip()) > 0:
    offre_text = offre_text_input.strip()
elif offre_file is not None:
    offre_text = offre_file.read().decode("utf-8")
else:
    offre_text = ""

col1, col2 = st.columns([2,1])

with col1:
    if st.button("Lancer l'analyse"):

        if len(uploaded_cv_files) == 0:
            st.warning("Téléverse au moins un fichier CV (.txt) pour analyser.")
        elif len(offre_text) == 0:
            st.warning("Fournis le texte de l'offre (colle-la ou téléverse le fichier).")
        else:
            # Lire tous les CVs
            names = []
            texts = []
            for f in uploaded_cv_files:
                try:
                    raw = f.read()
                    if isinstance(raw, bytes):
                        txt = raw.decode("utf-8", errors="ignore")
                    else:
                        txt = str(raw)
                except Exception as e:
                    txt = ""
                names.append(f.name)
                texts.append(txt)

            # Calculer scores
            with st.spinner("Encodage et calcul des similarités (Sentence-BERT)..."):
                scores = score_batch(texts, offre_text, model_name=model_name)

            # Construire DataFrame
            df = pd.DataFrame({
                "filename": names,
                "score": scores
            })
            df["result"] = df["score"].apply(lambda s: "MATCH" if s >= seuil else "PAS MATCH")
            st.success(f"Analyse terminée — {len(df)} CV(s) traités.")

            # Afficher tableau
            st.markdown("#### Résultats")
            st.dataframe(df.sort_values("score", ascending=False).reset_index(drop=True))

            # Téléchargement CSV
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger résultats (CSV)", data=csv_bytes, file_name="cv_matching_results.csv", mime="text/csv")

            # Histogramme des scores
            st.markdown("#### Distribution des scores")
            fig, ax = plt.subplots()
            ax.hist(df["score"], bins=10)
            ax.set_xlabel("Score de similarité")
            ax.set_ylabel("Nombre de CVs")
            ax.set_title("Histogramme des scores")
            st.pyplot(fig)

            # Afficher les meilleurs et pires
            st.markdown("#### Top 3 (meilleurs scores)")
            st.table(df.sort_values("score", ascending=False).head(3).reset_index(drop=True))

            st.markdown("#### Bottom 3 (pires scores)")
            st.table(df.sort_values("score", ascending=True).head(3).reset_index(drop=True))

with col2:
    st.markdown("### Notes & conseils")
    st.write("""
    - Le seuil par défaut est 0.50 ; ajuste en fonction de la sélectivité souhaitée.
    - Pour de meilleurs résultats, nettoie le texte des CV (enlever entêtes inutiles, sections vides).
    - Option avancée : utiliser un modèle custom ou fine-tuner sur un dataset étiqueté.
    """)
    st.markdown("### Exemple rapide")
    st.write("Ajoute 2-5 CVs simples et l'offre, puis clique sur *Lancer l'analyse*.")

# Footer (instructions)
st.markdown("---")
st.markdown("Projet — Classifieur CV (Sentence-BERT). À utiliser comme prototype pour un ATS simple.")
