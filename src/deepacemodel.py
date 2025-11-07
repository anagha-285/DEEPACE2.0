import torch
import torch.nn as nn

class DeepACEModel(nn.Module):
    """
    DeepACE2.0-inspired 1D CNN model for cochlear implant sound coding.
    Input: raw audio waveform
    Output: predicted electrodogram sequence
    """
    def __init__(self, num_electrodes=22):
        super(DeepACEModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.ConvTranspose1d(64, 32, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, num_electrodes, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.Sigmoid()  # Normalize outputs between 0–1 (stimulation levels)
        )

    def forward(self, x):
        # x shape: (batch, 1, samples)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
