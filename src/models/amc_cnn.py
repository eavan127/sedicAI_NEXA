"""
1D-CNN for Automatic Modulation Classification, trained from scratch.

There is no pretrained backbone for raw IQ the way ImageNet exists for
images — so "fine-tuning" here means training from random initialization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool1d(nn.Module):
    """Energy-gated attention pooling: learned weighted pooling over time,
    where the scorer sees both the learned conv features AND the raw signal's
    instantaneous power at each timestep -- instead of a plain average/max.

    A 1x1 conv scores every timestep from (conv features + local energy),
    softmax turns the scores into an attention distribution over time, and
    the output is the weighted sum of the conv features under that
    distribution.

    Why this over flatten()+Linear: flatten hands the final Linear layer one
    independent weight per absolute time position, learned once, globally,
    across the whole training set. That can't adapt to *where in this
    particular example* the signal actually is -- and radar's pulse start
    time is randomised per example (see config: time_delay_s), so the
    informative part of the window moves around from one example to the
    next. Attention pooling scores each example's own timeline and can learn
    to weight up wherever the pulse (or hop, or jammer) actually sits and
    weight down the noise-only stretches, rather than a fixed position.

    Why energy-gated specifically: the conv features alone give the scorer
    only what it has learned to extract, which may or may not include
    anything resembling "is there signal energy here." Concatenating raw
    instantaneous power (I^2 + Q^2) as an extra channel hands it that cue
    directly -- the same principle real radar systems use for energy
    detection (deciding signal-present by thresholding received power). This
    is power, not literally SNR -- true SNR needs a noise-floor estimate too,
    which no single timestep can supply -- but it is the closest thing
    actually computable per timestep, and it is a real, standard technique,
    not an invented one.

    Same mechanism also removes the fixed-input-length requirement: softmax
    + weighted sum works over any number of timesteps and always produces
    one vector of size `channels`, so this model has no input_len dependency
    in its weights at all -- unlike the old flatten-based version, whose
    Linear layer shape was tied to one specific window length.
    """

    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv1d(channels + 1, 1, kernel_size=1)

    def forward(self, x, raw_power):
        # x: (batch, channels, time_reduced)   raw_power: (batch, 1, time_raw)
        # Conv/pooling upstream changed the time axis length from the raw
        # input; adaptive pooling re-aligns the energy signal to whatever
        # length the conv features currently are, without hand-computing the
        # exact conv/pooling arithmetic.
        energy = F.adaptive_avg_pool1d(raw_power, x.shape[-1])
        combined = torch.cat([x, energy], dim=1)          # (batch, channels+1, time)
        weights = torch.softmax(self.score(combined), dim=2)  # (batch, 1, time)
        return (x * weights).sum(dim=2)                    # (batch, channels)


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
        # Instantaneous power at each raw sample -- I^2 + Q^2 -- the energy
        # cue AttentionPool1d gates on, computed once here from the original
        # input before any conv layer touches it.
        raw_power = x[:, 0:1, :] ** 2 + x[:, 1:2, :] ** 2   # (batch, 1, window_len)
        feats = self._features(x)
        x = self.attn_pool(feats, raw_power)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)
