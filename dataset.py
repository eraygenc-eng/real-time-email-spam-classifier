import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class SpamDataset(Dataset):
    def __init__(self, split="train", max_features=7000):
        # Read CSV file
        df = pd.read_csv("Data/norm_spam.csv")

        # Remove missing rows
        df = df.dropna()

        # Convert labels to numeric values
        df["label"] = df["label"].map({"norm":0, "spam":1})

        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split feature and label
        X = df["text"].values
        y = df["label"].values

        # Test, Train part
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        # Convert to text into TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=7000)

        X_train = vectorizer.fit_transform(X_train).toarray()
        X_val = vectorizer.transform(X_val).toarray()
        X_test = vectorizer.transform(X_test).toarray()

        # Data -> Tensor
        if split == "train":
            self.x = torch.tensor(X_train, dtype=torch.float32)
            self.y = torch.tensor(y_train, dtype=torch.float32).view(-1,1)

        elif split == "val":
            self.x = torch.tensor(X_val, dtype=torch.float32)
            self.y = torch.tensor(y_val, dtype=torch.float32).view(-1,1)

        elif split == "test":
            self.x = torch.tensor(X_test, dtype=torch.float32)
            self.y = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
        
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]