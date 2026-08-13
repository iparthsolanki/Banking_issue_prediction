from fastapi import FastAPI
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

# ============================
# Load Model & Supporting Files
# ============================

# Make sure these files exist inside the 'static' folder
model = tf.keras.models.load_model("static/BiGRU_model (1).keras")

with open("static/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("static/label_names.pkl", "rb") as f:
    label_names = pickle.load(f)

# Same max_len used during training
MAX_LEN = 30

# ============================
# Pydantic Request Schema
# ============================
class UserQuery(BaseModel):
    text: str

# ============================
# Text Preprocessing
# ============================
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# ============================
# Home Route
# ============================
@app.get("/")
def home():
    return {
        "message": "Banking77 Intent Prediction API is running successfully!"
    }

# ============================
# Prediction Route
# ============================
@app.post("/predict")
def predict(query: UserQuery):
    cleaned_text = preprocess_text(query.text)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Pad sequence
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # Model prediction
    prediction = model.predict(padded, verbose=0)

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "input_text": query.text,
        "predicted_intent": label_names[predicted_index],
        "confidence": round(confidence, 4)
    }