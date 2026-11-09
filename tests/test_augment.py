"""scripts/train_week1.py --augment: rotation by j^k and an optional conjugate,
checked against complex arithmetic."""
import numpy as np
import torch

from scripts.train_week1 import augment


def test_every_window_is_a_quarter_turn_of_itself_possibly_mirrored():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((64, 128)) + 1j * rng.standard_normal((64, 128))
    xb = torch.tensor(np.stack([z.real, z.imag], axis=1), dtype=torch.float32)
    out = augment(xb, torch.Generator().manual_seed(1)).numpy()
    w = out[:, 0] + 1j * out[:, 1]
    seen = set()
    for a, b in zip(z, w):
        match = [(k, c) for k in range(4) for c in (False, True)
                 if np.allclose(np.conj(a * 1j ** k) if c else a * 1j ** k, b, atol=1e-5)]
        assert match, "augmented window is not a rotation/flip of the original"
        seen.add(match[0])
    assert len(seen) == 8                       # all four rotations, with and without the flip


def test_power_and_shape_are_unchanged():
    xb = torch.randn(32, 2, 512)
    out = augment(xb)
    assert out.shape == xb.shape
    assert torch.allclose((out ** 2).sum((1, 2)), (xb ** 2).sum((1, 2)), atol=1e-4)
