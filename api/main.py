from fastapi import FastAPI, HTTPException, Body
from nltk.corpus import stopwords
from pydantic import BaseModel, Field, field_validator
import joblib
import tensorflow as tf
import contractions
import spacy
import re
import os
import numpy as np


app = FastAPI(
    title= "Fake news Detector API",
    description= "Prediction of Fake News and Real News",
    version= "1.0.0"
)

class PredictionRequest(BaseModel):
    title: str = Field(..., description= "Article title")

    @ field_validator('title')
    @classmethod
    def validate_title(cls, v) : 
        if not v.strip() :
            raise HTTPException(status_code= 422, detail =" Le titre ne peut pas être vide ou composé uniquement d'espaces" )
        if len(v) > 300 :
            raise HTTPException(status_code= 400, detail = "Le titre est trop long, il ne doit pas dépasser 300 caractères")
        return v


class BatchRequest(BaseModel): 
    titles: list[str] = Field(..., description= "Article title list")

    @field_validator('titles')
    @classmethod
    def validate_batch(cls, v) :
        if len(v) == 0 :
            raise HTTPException(status_code= 422, detail = "Un des titres est vide ou composé d'espaces")
        if len(v) > 50 :
            raise HTTPException(status_code= 400, detail = "La liste ne dois pas dépasser les 50 titres")
        
        for t in v : 
            if not t.strip() :
                raise HTTPException(status_code = 422, detail = "Un des titres est vide ou composé d'espaces")
            if len(t) > 300 :
                raise HTTPException(status_code = 400, detail = " Un des titres dépasse les 300 caractères")
        return v


model = None
vectorizer = None
nlp = spacy.load("en_core_web_sm", disable= ["parser", "ner"])

def clean_title(text: str) -> str :
    # mise en minuscule
    text = text.lower()

    # Suppresion des urls et des mentions de type '@username'
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)

    # ponctuaction
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Expansion des contractions (installation des contractions)
    text = contractions.fix(text)

    # Stopwords
    stop_words = set(stopwords.words('english'))
    word_negations = {'not', 'no', 'never', 'neither'}
    filtered_stops = stop_words - word_negations

    # Lemmatisation
    doc = nlp(text)

    # Suppression des tokens < 2 après lemmatisation
    cleaned_tokens = [
        token.lemma_ for token in doc 
        if token.lemma_ not in filtered_stops 
        and not token.is_space 
        and len(token.lemma_) > 1
    ]

    return " ".join(cleaned_tokens)



@app.on_event("startup")
async def load_model():
    global model, vectorizer
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, "models", "best_models_tfidf.keras")
    vectorizer_path = os.path.join(base_path, "models", "vectorizer.pkl")
    
    try:
        model = tf.keras.models.load_model(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Modèle et vectoriseur chargés avec succès !")
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")



@app.get("/health")
def health_check():
    return {"status": "ok", "model": "fake_news_detector"}


@app.post("/predict")
def predict(request: PredictionRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    cleaned_title = clean_title(request.title)
    vectorized_title = vectorizer.transform([cleaned_title]).toarray()
    
    # Prédiction
    prob = float(model.predict(vectorized_title)[0][0])
    label = "REAL" if prob >= 0.5 else "FAKE"
    confidence = prob if prob >= 0.5 else (1 - prob)
    
    return {
        "title": request.title,
        "label": label,
        "confidence": round(confidence, 4)
    }


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    cleaned_titles = [clean_title(t) for t in request.titles]
    vectorized_titles = vectorizer.transform(cleaned_titles).toarray()
    
    probs = model.predict(vectorized_titles).flatten()
    results = []
    
    for title, prob in zip(request.titles, probs):
        label = "REAL" if prob >= 0.5 else "FAKE"
        conf = float(prob if prob >= 0.5 else (1 - prob))
        results.append({
            "title": title,
            "label": label,
            "confidence": round(conf, 4)
        })
        
    return {"predictions": results}