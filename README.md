## Contexte

Un organisme de vérification des faits (fact-checking) publie chaque semaine des rapports sur la fiabilité des articles de presse en ligne. Il reçoit des milliers de titres d'articles issus de sources variées — certains sont des titres d'articles journalistiques vérifiés, d'autres sont des titres de contenus identifiés comme trompeurs ou sensationnalistes.

L'organisme souhaite automatiser un premier niveau de triage : classer chaque titre entrant comme **fiable** ou **trompeur**, avant qu'un analyste humain ne prenne la décision finale. Ce pré-filtrage doit diviser par trois la charge de travail manuelle.

Vous êtes développeur IA missionné pour concevoir et implémenter ce pipeline NLP de bout en bout, depuis le chargement des données jusqu'à l'exposition du modèle via une API de prédiction.


## Installation et lancement
1. Prérequis
- Python 3.10+
- Les dépendances listées dans requirements.txt

2. Installation

Pour l'instalation des bibliothèques : 

``` python
pip install -r requirements.txt
```


Pour le téléchargement du modèle linguistique spacy : 
``` python
python -m spacy download en_core_web_sm
```


3. Lancer l'API 
``` python
python -m uvicorn api.main:app --reload
```

L'interface Swagger est maintenant accessible sur : http://127.0.0.1:8000/docs


## Architecture du projet
- notebooks/ : Etude exploratoire (EDA) et entraînement du modèle.
- api/main.py : Serveur FastAPI gérant les prédictions unitaires en lot. 
- models/ : Contient le modèle keras et le vectoriseur TF-IDF sauvegardés. 
- data/ : Dataset d'origine (fake_or_real_news.csv) et dataset nettoyé (titles_clean.csv)

C:.
│   .gitignore
│   ECF_NLP_TF_FakeNews.md
│   README.md
│   requirements.txt
│   
├───api
│   │   main.py
│   │   
│   └───__pycache__
│           main.cpython-311.pyc
│           
├───data
│       fake_or_real_news.csv
│       titles_clean.csv
│       
├───models
│       best_models_tfidf.keras
│       vectorizer.pkl
│       
└───notebook
    │   ecf_fake_news.ipynb
    │   
    └───data


## Performance et analyse du modèle
1. Caractéristiques techniques 
- Algotithme : Réseau de neurones Dense (Deep Learning) avec couches Dropout pour limiter le sur-apprentissage (Overfiting)
- Vectorisation : TF-IDF (Team Frequency-Inverse Document Frequency) limité aux 5000 termes les plus fréquents
- Prétraitement : Nettoyage par Regex, expansion des contractions, suppression des stop-words  et lemmatisation via Spacy. 



2. Résultats obtenus
- Accuracy (test) : 81,93% (C'est ce chiffre qu'il faut retenir pour la performance globale).
- Précision (classe FAKE) : 80,22% (Capacité à ne pas se tromper quand il prédit un Fake).
- Recall (classe FAKE) : 83,71% (Capacité à détecter tous les Fakes présents).
- AUC-ROC : 0,8889 (Indique une très bonne capacité de séparation des classes par le modèle).
- Analyse de robustesse : Le modèle performe très bien sur les thématiques politiques (sujet principal du dataset d'entraînement). Cependant, il présente des faiblesses sur les actualités généralistes (science, économie) où il a tendance à sur-prédire la classe FAKE, soulignant une dépendant vocabulaire spécifique de 2016.


## Validation & Cas Limites (API)
L'API intègre des contrôles de sécurité et de conformité pour assurer la stabilité du service :
- Validation des titres : Les titres vides ou composés uniquement d'espaces sont rejetés (Erreur 422).
- Contrôle de longueur : Les titres dépassant 300 caractères retournent une erreur 400 pour éviter les dépassements de mémoire.
- Gestion du Batch : L'envoi groupé est limité à 50 titres maximum par requête pour garantir un temps de réponse optimal.


## Utilisation de l'API
| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Vérifie l'état de l'API du modèle |
| `POST` | `/predict` | Prédiction unitaire pour un titre |
| `POST` | `/predict/batch` | Prédiction en lot (liste de titres) |