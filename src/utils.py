import torch
from torch.utils.data import Dataset
import scipy.io as sio
import os
import numpy as np
from sklearn.model_selection import train_test_split

class EEGDataset(Dataset):
    """Dataset for DeepACE2.0 .mat files (x: audio, y: electrodogram)."""
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.y_data[idx], dtype=torch.float32)
        return x, y

def load_data(folder):
    """Load .mat files from folder and split into train/val sets."""
    x_list, y_list = [], []
    for file in os.listdir(folder):
        if file.endswith(".mat"):
            data = sio.loadmat(os.path.join(folder, file))
            if "x" in data and "y" in data:
                x_list.append(data["x"].squeeze())
                y_list.append(data["y"])

    x_array = np.array(x_list, dtype=object)
    y_array = np.array(y_list, dtype=object)
    x_train, x_val, y_train, y_val = train_test_split(x_array, y_array, test_size=0.2, random_state=42)

    train_dataset = EEGDataset(x_train, y_train)
    val_dataset = EEGDataset(x_val, y_val)
    return train_dataset, val_dataset
