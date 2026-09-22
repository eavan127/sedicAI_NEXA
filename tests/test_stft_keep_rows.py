"""
Tests for the experimental `model.stft_keep_rows` option on STFTBranch (variant "C2").

STFTBranch used to end with `f.mean(dim=2)`, which averages the frequency rows away, so
"a wide jammer block on rows 1-3 AND a hopper dash on row 6" and "a slightly louder jammer"
become nearly the same number. With `stft_keep_rows` the (detectors x rows) block is laid out
as one long channel axis and a learned 1x1 layer decides what each (detector, row) pair is
worth. Output is still `out_channels` wide, so fusion and the head are untouched.

These tests protect three things:
  1. Flag OFF is exactly today's architecture -- same parameters, same state_dict keys, so
     every checkpoint in results/ keeps loading.
  2. Flag ON really keeps the rows: two blocks with the SAME row average but different row
     positions must give different outputs (and must give the SAME output with the flag off).
  3. The exported (ONNX) copy of the branch computes the same network, because the browser
     demo runs that copy, not this module.
"""
import pytest
import torch

from src.config import CFG, CLASSES
from src.models.amc_cnn import AMC_CNN, STFTBranch
from src.models.onnx_export import AMC_CNN_ONNX, compute_stft_mag

WINDOW_LEN = 512
C = 64          # detectors (STFTBranch out_channels)
ROWS = 8        # frequency rows left after the default 2x2 pool (16 // 2)


def _iq(batch=3):
    g = torch.Generator().manual_seed(0)
    return torch.randn(batch, 2, WINDOW_LEN, generator=g)


def _n_params(module):
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# 1. Flag off == today's behaviour.
# ---------------------------------------------------------------------------

def test_keep_rows_is_off_in_the_config():
    assert CFG.get("model", {}).get("stft_keep_rows") is False


def test_flag_off_builds_no_new_layers_and_the_state_dict_keys_are_unchanged():
    off = STFTBranch()
    assert off.keep_rows is False
    assert not hasattr(off, "row_proj") and not hasattr(off, "row_bn")
    assert not any(k.startswith("row_") for k in off.state_dict())


def test_flag_off_parameter_count_is_the_published_one():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                    stft_freq_summary=False, stft_keep_rows=False)
    assert _n_params(model) == 148_938          # the number in the report (section 4.1)


# ---------------------------------------------------------------------------
# 2. Flag on: shapes and cost.
# ---------------------------------------------------------------------------

def test_flag_on_output_shape_matches_flag_off():
    x = _iq()
    off, on = STFTBranch(), STFTBranch(keep_rows=True)
    assert on(x).shape == off(x).shape             # (batch, 64, time') either way


def test_flag_on_cost_is_exactly_the_learned_layer_and_its_batchnorm():
    off, on = STFTBranch(), STFTBranch(keep_rows=True)
    learned_layer = C * ROWS * C + C               # Conv1d(512 -> 64, k=1) weight + bias
    batchnorm = 2 * C                              # BatchNorm1d weight + bias
    assert _n_params(on) - _n_params(off) == learned_layer + batchnorm


def test_fusion_and_head_widths_are_unchanged():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN, stft_keep_rows=True)
    assert model.fc1.in_features == 128 + 64
    assert model(_iq()).shape == (3, len(CLASSES))


def test_keep_rows_composes_with_freq_summary_and_uses_all_sixteen_rows():
    both = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                   stft_freq_summary=True, stft_keep_rows=True)
    assert both.stft_branch.row_proj.in_channels == C * 16     # time-only pool keeps 16 rows
    assert both.fc1.in_features == 128 + 64 + 3                # plus the three side channels
    assert torch.isfinite(both(_iq())).all()


# ---------------------------------------------------------------------------
# 3. It really keeps the rows.
# ---------------------------------------------------------------------------

def _block(row_values):
    """A (1, C, ROWS, 1) block that looks the same to every detector: `row_values`
    across the 8 frequency rows."""
    v = torch.tensor(row_values, dtype=torch.float32)
    return v.view(1, 1, ROWS, 1).expand(1, C, ROWS, 1).contiguous()


def test_averaging_cannot_tell_a_jammer_block_from_a_lone_hopper_dash():
    """The toy example from the explanation: same row average, different rows."""
    x_block = _block([0, 0, 3, 3, 3, 0, 0, 0])      # wide block, average 9/8
    y_dash = _block([0, 0, 0, 0, 0, 0, 9, 0])       # one thin dash, average 9/8
    branch = STFTBranch(keep_rows=False).eval()
    assert torch.allclose(branch._collapse_rows(x_block), branch._collapse_rows(y_dash))


def test_keep_rows_tells_them_apart():
    x_block = _block([0, 0, 3, 3, 3, 0, 0, 0])
    y_dash = _block([0, 0, 0, 0, 0, 0, 9, 0])
    torch.manual_seed(0)
    branch = STFTBranch(keep_rows=True).eval()
    with torch.no_grad():
        a, b = branch._collapse_rows(x_block), branch._collapse_rows(y_dash)
    assert not torch.allclose(a, b, atol=1e-4)


def test_moving_a_pattern_to_another_row_changes_the_output_only_with_keep_rows():
    """Same energy, same shape, different row."""
    low = _block([0, 5, 0, 0, 0, 0, 0, 0])
    high = _block([0, 0, 0, 0, 0, 0, 5, 0])
    torch.manual_seed(1)
    kept = STFTBranch(keep_rows=True).eval()
    averaged = STFTBranch(keep_rows=False).eval()
    with torch.no_grad():
        assert torch.allclose(averaged._collapse_rows(low), averaged._collapse_rows(high))
        assert not torch.allclose(kept._collapse_rows(low), kept._collapse_rows(high), atol=1e-4)


def test_forward_is_conv_path_then_the_learned_layer_in_frequency_order():
    """The whole branch equals: shifted magnitude -> conv1/pool/conv2 -> flatten -> row_proj."""
    torch.manual_seed(2)
    branch = STFTBranch(keep_rows=True).eval()
    x = _iq(batch=2)
    with torch.no_grad():
        spec = torch.stft(torch.complex(x[:, 0], x[:, 1]), n_fft=branch.n_fft,
                          hop_length=branch.hop_length, window=branch.window,
                          center=False, return_complex=True)
        mag = torch.fft.fftshift(spec.abs().unsqueeze(1), dim=2)
        f = branch.pool(branch.relu(branch.bn1(branch.conv1(mag))))
        f = branch.relu(branch.bn2(branch.conv2(f)))
        expected = branch.relu(branch.row_bn(branch.row_proj(f.flatten(1, 2))))
        assert torch.allclose(branch(x), expected, atol=1e-6)


def test_rows_reach_conv1_in_frequency_order_when_keep_rows_is_on():
    """Row 0 must be the LOWEST frequency, not 0 Hz. A tone at 0 Hz must sit in the middle."""
    n = WINDOW_LEN
    zero_hz = torch.stack([torch.ones(n), torch.zeros(n)]).unsqueeze(0)     # a constant = 0 Hz
    seen = {}

    def grab(_module, inputs):
        seen["mag"] = inputs[0].detach()

    for keep in (True, False):
        branch = STFTBranch(keep_rows=keep).eval()
        handle = branch.conv1.register_forward_pre_hook(grab)
        branch(zero_hz)
        handle.remove()
        strongest_row = int(seen["mag"][0, 0].mean(dim=1).argmax())
        # shifted: 0 Hz is row n_fft//2 = 8.  raw FFT order: 0 Hz is row 0.
        assert strongest_row == (8 if keep else 0)


def test_gradients_reach_the_learned_layer():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN, stft_keep_rows=True)
    model(_iq()).sum().backward()
    g = model.stft_branch.row_proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


# ---------------------------------------------------------------------------
# 4. The exported copy computes the same network.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(),                                                   # default, both flags off
    dict(stft_freq_summary=True),                             # the earlier flag (row-order shift)
    dict(stft_keep_rows=True),                                # C2
    dict(stft_keep_rows=True, stft_freq_summary=True),        # both
])
def test_onnx_wrapper_matches_the_model(kwargs):
    torch.manual_seed(3)
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN, **kwargs).eval()
    x = _iq(batch=2)
    with torch.no_grad():
        mag = compute_stft_mag(x, n_fft=model.stft_branch.n_fft,
                               hop_length=model.stft_branch.hop_length,
                               window=model.stft_branch.window)
        logits, _attention = AMC_CNN_ONNX(model)(x, mag)
        assert torch.allclose(logits, model(x), atol=1e-5)
