from model import NeuralNetwork
from dataset import SpamDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device: ", device)

# Dataset
train_dataset = SpamDataset(split="train")
val_dataset = SpamDataset(split="val")

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model
model = NeuralNetwork(input_dim=7000)
model.to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# Accuracy
def calculate_acc(logits, y_true):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    correct = (preds == y_true).sum().item()
    accuracy = correct / y_true.size(0)
    return accuracy

# Early Stopping Settings
epochs = 20
patience = 5
best_val_loss = float("inf")
counter = 0

# Training + Validation
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_acc = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(x_batch)

        loss = criterion(logits, y_batch)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        train_acc += calculate_acc(logits, y_batch)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            val_loss += loss.item()
            val_acc += calculate_acc(logits, y_batch)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

# Best Model Save
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        print(f"no improvement: {counter}/{patience}")

    # Early Stopping
    if counter == patience:
        print("Early stopped")
        break

    print(
        f"Epoch {int(epoch)+1}/{int(epochs)} | "
        f"Train Loss: {float(train_loss):.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {float(val_loss):.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

joblib.dump(train_dataset.vectorizer, "vectorizer.pkl")
print("Vectorizer saved.")