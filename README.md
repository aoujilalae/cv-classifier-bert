# Classifieur de CV — Match / Pas Match (Sentence-BERT)
## Version améliorée (Batch, Streamlit, Export CSV, Visualisation)
### Fonctionnalités
- Upload multiple CVs (.txt)
- Coller ou upload d'une offre
- Calcul automatique de similarité (Sentence-BERT)
- Seuil ajustable pour décider MATCH / PAS MATCH
- Export CSV des résultats
- Histogramme des scores
### Installation rapide
```bash
# (optionnel) créer et activer un virtualenv
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# Installer dépendances
pip install -r requirements.txt