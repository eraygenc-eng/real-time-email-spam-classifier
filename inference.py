import torch
import joblib
from model import NeuralNetwork

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Load Model
model = NeuralNetwork(input_dim=7000)
model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

def predict(text):
    x = vectorizer.transform([text]).toarray()
    x = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)

    spam_prob = prob.item()

    if spam_prob >= 0.5:
        pred = "spam"
    else:
        pred = "ham"
    return pred, spam_prob

# Test
text = input("Enter mail text: ")

prediction, probability = predict(text)

print("Prediction:", prediction)
print(f"Spam probability: {probability:.4f}")