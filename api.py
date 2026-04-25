import torch
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from model import NeuralNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vectorizer = joblib.load("vectorizer.pkl")

model = NeuralNetwork(input_dim=7000)
model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()


class MailRequest(BaseModel):
    text: str


app = FastAPI()


def predict(text):
    x = vectorizer.transform([text]).toarray()
    x = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    prediction = "spam" if prob >= 0.5 else "norm"

    return prediction, prob


@app.get("/")
def home():
    return {"message": "Mail Spam Classifier API is running"}


@app.post("/predict")
def predict_mail(mail: MailRequest):
    prediction, probability = predict(mail.text)

    return {
        "prediction": prediction,
        "spam_probability": round(probability, 4)
    }