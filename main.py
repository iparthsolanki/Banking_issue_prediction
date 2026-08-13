from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences



# ============================
# FastAPI App
# ============================
app = FastAPI(
    title="Banking77 Intent Prediction API",
    description="Predict banking intent using a trained BiGRU model",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/app")
def frontend():
    return FileResponse("frontend/index.html")
# ============================
# Load Model & Files
# ============================
model = tf.keras.models.load_model("static/BiGRU_model.keras")

with open("static/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("static/label_names.pkl", "rb") as f:
    label_names = pickle.load(f)

MAX_LEN = 30

# ============================
# Pydantic Schema
# ============================
class UserQuery(BaseModel):
    text: str

# ============================
# Preprocessing
# ============================
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# ============================
# Routes
# ============================
@app.get("/")
def home():
    return {"message": "Banking77 Intent Prediction API is running successfully!"}

@app.post("/predict")
def predict(query: UserQuery):
    cleaned = preprocess_text(query.text)

    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded, verbose=0)
    idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "input_text": query.text,
        "predicted_intent": label_names[idx],
        "confidence": round(confidence, 4)
    }