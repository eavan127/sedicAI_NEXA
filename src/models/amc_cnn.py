"""
1D-CNN for Automatic Modulation Classification, trained from scratch.

There is no pretrained backbone for raw IQ the way ImageNet exists for
images — so "fine-tuning" here means training from random initialization.
"""
import torch
import torch.nn as nn


class AttentionPool1d(nn.Module):
    """Learned weighted pooling over time, instead of a plain average/max.

    A 1x1 conv scores every timestep, softmax turns the scores into an
    attention distribution over time, and the output is the weighted sum of
    the features under that distribution.

    Why this over flatten()+Linear: flatten hands the final Linear layer one
    independent weight per absolute time position, learned once, globally,
    across the whole training set. That can't adapt to *where in this
    particular example* the signal actually is -- and radar's pulse start
    time is randomised per example (see config: time_delay_s), so the
    informative part of the window moves around from one example to the
    next. Attention pooling scores each example's own timeline and can learn
    to weight up wherever the pulse (or hop, or jammer) actually sits and
    weight down the noise-only stretches, rather than a fixed position.

    Same mechanism also removes the fixed-input-length requirement: softmax
    + weighted sum works over any number of timesteps and always produces
    one vector of size `channels`, so this model has no input_len dependency
    in its weights at all -- unlike the old flatten-based version, whose
    Linear layer shape was tied to one specific window length.
    """

    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (batch, channels, time)
        weights = torch.softmax(self.score(x), dim=2)   # (batch, 1, time)
        return (x * weights).sum(dim=2)                  # (batch, channels)


class AMC_CNN(nn.Module):
    def __init__(self, num_classes, input_len=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.attn_pool = AttentionPool1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # No dummy forward pass to size this anymore -- attention pooling
        # always outputs exactly `channels` (128) values regardless of
        # input_len, so fc1's input size no longer depends on window length
        # at all. input_len is kept as a parameter only so existing call
        # sites (AMC_CNN(num_classes=..., input_len=X.shape[-1])) still work
        # unchanged; it is otherwise unused.
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _features(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

    def forward(self, x):
        x = self.attn_pool(self._features(x))
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)
