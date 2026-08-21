"""
Pipeline shape/wiring tests — cheap guards that catch a broken refactor before
someone burns an hour of GPU time discovering it.
"""
import numpy as np
import pytest
import torch

from src.config import CFG, CLASSES
from src.models.amc_cnn import AMC_CNN
from src.train import compute_class_weights, stratified_split


def _one_hot(class_indices, num_classes):
    """Turn a flat list of class indices into a (N, num_classes) multi-hot
    array, one bit per row -- the shape stratified_split/compute_class_weights
    now expect, since a row can in principle have more than one bit set for a
    composite example (unused by these tests, which only need standalone
    rows)."""
    y = np.zeros((len(class_indices), num_classes))
    y[np.arange(len(class_indices)), class_indices] = 1
    return y


class TestModel:
    def test_output_shape(self):
        model = AMC_CNN(num_classes=len(CLASSES), input_len=1024)
        out = model(torch.zeros(4, 2, 1024))
        assert out.shape == (4, len(CLASSES))

    @pytest.mark.parametrize("input_len", [512, 1024, 2048])
    def test_adapts_to_window_length(self, input_len):
        """Flattened width is inferred, not hardcoded — changing window_len in
        the config must not silently break the model."""
        model = AMC_CNN(num_classes=len(CLASSES), input_len=input_len)
        assert model(torch.zeros(2, 2, input_len)).shape == (2, len(CLASSES))

    def test_gradients_flow(self):
        """Uses BCEWithLogitsLoss with multi-hot float targets -- the actual
        production loss (src/train.py), now that classes are judged
        independently rather than via a single argmax winner."""
        model = AMC_CNN(num_classes=len(CLASSES), input_len=1024)
        targets = (torch.rand(8, len(CLASSES)) > 0.5).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(torch.randn(8, 2, 1024)), targets
        )
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)


class TestSplit:
    def _fixture(self):
        labels, snr = [], []
        for cls in range(len(CLASSES)):
            for s in CFG["snr_bins_db"]:
                labels += [cls] * 20
                snr += [s] * 20
        return _one_hot(labels, len(CLASSES)), np.array(snr, dtype=float)

    def test_splits_are_disjoint_and_complete(self):
        y, snr = self._fixture()
        tr, va, te = stratified_split(y, snr, 0.15, 0.15, seed=42)
        assert len(set(tr) & set(va)) == 0
        assert len(set(tr) & set(te)) == 0
        assert len(set(va) & set(te)) == 0
        assert len(tr) + len(va) + len(te) == len(y)

    def test_every_class_and_snr_appears_in_test_split(self):
        """Without this, the accuracy-vs-SNR curve has holes and the per-class
        recall benchmark cannot be computed for a missing class."""
        y, snr = self._fixture()
        _, _, te = stratified_split(y, snr, 0.15, 0.15, seed=42)
        assert set(y.argmax(axis=1)[te]) == set(y.argmax(axis=1))
        assert set(np.unique(snr[te])) == set(np.unique(snr))

    def test_deterministic_for_a_given_seed(self):
        y, snr = self._fixture()
        a = stratified_split(y, snr, 0.15, 0.15, seed=42)
        b = stratified_split(y, snr, 0.15, 0.15, seed=42)
        assert all(np.array_equal(x, z) for x, z in zip(a, b))


class TestClassWeights:
    def test_rare_classes_weigh_more(self):
        """The judged classes are the minority; if weighting ever inverts, the
        model quietly optimises for the classes we are not scored on."""
        y = _one_hot([0] * 900 + [1] * 100, 2)
        w = compute_class_weights(y, 2)
        assert w[1] > w[0]

    def test_handles_absent_class_without_dividing_by_zero(self):
        w = compute_class_weights(_one_hot([0, 0, 1], 3), num_classes=3)
        assert torch.isfinite(w).all()
