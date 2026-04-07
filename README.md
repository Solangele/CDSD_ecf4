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

