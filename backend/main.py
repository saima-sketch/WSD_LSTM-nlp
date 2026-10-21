"""
main.py
-------
FastAPI backend for the WSD LSTM application.
Exposes:
    GET  /         -> status
    POST /predict  -> Word Sense Disambiguation
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the model and prediction function.
# Training happens once when this module is imported.
from model import predict_sense

app = FastAPI(title="WSD LSTM API", version="1.0")

# ------------------------------------------------------------------
# CORS: allow the frontend served at http://127.0.0.1:5500
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Request model
# ------------------------------------------------------------------
class SentenceRequest(BaseModel):
    sentence: str

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/")
def read_root():
    """Health check / status message."""
    return {
        "status": "ok",
        "message": "WSD LSTM API is running. Use POST /predict with {'sentence': '...'}"
    }

@app.post("/predict")
def predict(request: SentenceRequest):
    """
    Receive a sentence and return the predicted sense of 'bank'.
    """
    result = predict_sense(request.sentence)
    return {
        "sentence": request.sentence,
        "class": result["class"],
        "meaning": result["meaning"],
        "probabilities": result["probabilities"]
    }