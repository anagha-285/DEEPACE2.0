import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import EEGDataset, load_data
from deepace_model import DeepACEModel
from tqdm import tqdm
import os

# Hyperparameters
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_ELECTRODES = 22

# Paths
DATA_DIR = "data/nmt_preprocessed"
MODEL_SAVE_PATH = "results/models/deepace_model.pt"

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Training on: {device}")

    # Load dataset
    train_data, val_data = load_data(DATA_DIR)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # Model setup
    model = DeepACEModel(num_electrodes=NUM_ELECTRODES).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Training Loss: {avg_train_loss:.6f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.6f}")

        # Save best model
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
