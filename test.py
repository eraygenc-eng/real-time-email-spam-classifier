import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SpamDataset
from model import NeuralNetwork

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset and Dataloader
test_dataset = SpamDataset(split="test")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
model = NeuralNetwork(input_dim=7000)
model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

# Loss
criterion = nn.BCEWithLogitsLoss()

# Accuracy
def calculate_acc(logits, y_true):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    correct = (preds == y_true).sum().item()
    accuracy = correct / y_true.size(0)
    return accuracy

# Test
test_loss = 0
test_acc = 0

with torch.no_grad():
     for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        test_loss += loss.item()
        test_acc += calculate_acc(logits, y_batch)

test_loss /= len(test_loader)
test_acc /= len(test_loader)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")