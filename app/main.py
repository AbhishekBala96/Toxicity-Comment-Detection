from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
import keras 
import tensorflow as tf

# CONFIG
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR/"toxic_classifier_model.keras"
TOKENIZER_PK = BASE_DIR/"tokenizer.pkl"
LABELS_PK = BASE_DIR/"labels.pkl"
MAX_SEQUENCE_LENGTH = 200

#LOAD
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
      epsilon = K.epsilon()
      y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

      alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
      p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

      focal_weight = alpha_t * K.pow((1 - p_t), gamma)
      focal_loss = -focal_weight * K.log(p_t + epsilon)

      return K.mean(focal_loss)
    return focal_loss_fixed

model = load_model(MODEL_PATH,
                   custom_objects={"focal_loss_fixed": focal_loss})
with open(TOKENIZER_PK, "rb") as f:
    tokenizer = pickle.load(f)
with open(LABELS_PK, "rb") as f:
    label_cols = pickle.load(f)

# APP

app = FastAPI(title = "Toxic Comment Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.get("/healthz")
def health():
    return {"status": "ok", "labels": label_cols}

@app.post("/predict")
def predict_toxicity(payload: TextInput):
    seq = tokenizer.texts_to_sequences([payload.text])
    x = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    probs = model.predict(x, verbose=0)[0]
    preds = {label: float(p) for label, p in zip(label_cols, probs)}
    return {"predictions": preds}