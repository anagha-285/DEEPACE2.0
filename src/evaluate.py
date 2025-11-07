import torch
import matplotlib.pyplot as plt
from deepace_model import DeepACEModel
from utils import load_data
import numpy as np

MODEL_PATH = "results/models/deepace_model.pt"
DATA_DIR = "data/nmt_preprocessed"

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepACEModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    _, val_data = load_data(DATA_DIR)
    x, y = val_data[0]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).cpu().numpy().squeeze()
        target = y.numpy().squeeze()

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.imshow(pred, aspect='auto', origin='lower')
    plt.title("Predicted Electrodogram")
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(target, aspect='auto', origin='lower')
    plt.title("Target Electrodogram")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
