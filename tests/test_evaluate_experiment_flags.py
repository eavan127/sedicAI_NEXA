"""
Which architecture evaluate_experiment builds for a checkpoint.

Getting this wrong does not produce a slightly-off number: the state_dict
fails to load and the run dies partway through, after the data has been read
and the val split predicted. It happened for real -- a cumulant-trained
checkpoint could not be scored at all, because the script had no
--cumulant-features flag and defaulted every flag to off.

run_stft_experiment.py writes experiment_<tag>_config.json beside each
checkpoint, so the flags are recoverable without anyone retyping them.
"""
import json
import types

import pytest

from scripts.evaluate_experiment import MODEL_FLAGS, resolve_flags

ALL_OFF = {"stft_freq_summary": False, "stft_keep_rows": False, "cumulant_features": False}


def args(**over):
    return types.SimpleNamespace(**{**ALL_OFF, **over})


def ckpt(tmp_path, name, flags=None, **extra):
    """A checkpoint path, with its sidecar when `flags` is given."""
    path = tmp_path / f"{name}.pt"
    path.write_bytes(b"")
    if flags is not None:
        (tmp_path / f"{name}_config.json").write_text(json.dumps({**flags, **extra}))
    return path


def test_flags_given_on_the_command_line_win(tmp_path):
    # The sidecar says one thing, the operator says another: the operator wins,
    # so a mislabelled sidecar can always be overridden.
    p = ckpt(tmp_path, "experiment_rows", {**ALL_OFF, "stft_keep_rows": True})
    assert resolve_flags([p], args(cumulant_features=True)) == {
        **ALL_OFF, "cumulant_features": True}


def test_flags_are_read_from_the_sidecar_when_none_are_given(tmp_path):
    p = ckpt(tmp_path, "experiment_freq_cumulant",
             {"stft_freq_summary": True, "stft_keep_rows": False, "cumulant_features": True},
             seed=2000, parameters=149709)
    assert resolve_flags([p], args()) == {
        "stft_freq_summary": True, "stft_keep_rows": False, "cumulant_features": True}


def test_a_checkpoint_without_a_sidecar_stays_all_off(tmp_path):
    # results/ensemble_*.pt were trained before the flags existed and have no
    # sidecar; they must keep scoring as the plain architecture.
    assert resolve_flags([ckpt(tmp_path, "ensemble_0")], args()) == ALL_OFF


def test_several_checkpoints_agreeing_is_fine(tmp_path):
    flags = {**ALL_OFF, "stft_keep_rows": True}
    paths = [ckpt(tmp_path, f"experiment_rows_seed{i}", flags) for i in range(3)]
    assert resolve_flags(paths, args()) == flags


def test_a_mixed_set_is_refused_rather_than_averaged(tmp_path):
    # Averaging two architectures is not a model. The old code would have built
    # whichever came first and failed later with a missing-key error.
    a = ckpt(tmp_path, "experiment_rows", {**ALL_OFF, "stft_keep_rows": True})
    b = ckpt(tmp_path, "experiment_cumulant", {**ALL_OFF, "cumulant_features": True})
    with pytest.raises(AssertionError, match="different architectures"):
        resolve_flags([a, b], args())


def test_a_sidecar_missing_a_key_reads_as_off(tmp_path):
    # Sidecars written before a flag existed simply do not mention it.
    p = ckpt(tmp_path, "experiment_old", {"stft_freq_summary": True})
    assert resolve_flags([p], args()) == {**ALL_OFF, "stft_freq_summary": True}


def test_every_model_flag_is_covered(tmp_path):
    # A flag added to the model but not to MODEL_FLAGS would silently stop
    # being carried from the sidecar.
    from src.models.amc_cnn import AMC_CNN
    import inspect

    params = set(inspect.signature(AMC_CNN.__init__).parameters) - {
        "self", "num_classes", "input_len"}
    assert params == set(MODEL_FLAGS), f"model takes {sorted(params)}, resolver knows {sorted(MODEL_FLAGS)}"
