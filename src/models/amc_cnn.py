"""
1D-CNN for Automatic Modulation Classification, trained from scratch.

There is no pretrained backbone for raw IQ the way ImageNet exists for
images — so "fine-tuning" here means training from random initialization.
"""
import torch
import torch.nn as nn


class AMC_CNN(nn.Module):
    def __init__(self, num_classes, input_len=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Infer the flattened width from a dummy pass rather than hand-computing
        # it — padding/pooling arithmetic is easy to get subtly wrong.
        with torch.no_grad():
            flat_len = self._features(torch.zeros(1, 2, input_len)).flatten(1).shape[1]

        self.fc1 = nn.Linear(flat_len, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _features(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        return x

    def forward(self, x):
        x = self._features(x).flatten(1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)
