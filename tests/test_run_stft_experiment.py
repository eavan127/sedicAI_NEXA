"""The experiment runner and the Colab notebook must agree on file names, and two runs must never
share one. A mismatch here would train for hours and then evaluate the wrong (or a missing) file."""
from scripts.run_stft_experiment import run_tag


def test_the_default_run_keeps_its_historical_name():
    assert run_tag(True, False, 20) == "stft_freq_summary"


def test_every_cell_of_the_plan_gets_its_own_name():
    cells = [(False, False, 40), (True, False, 20), (False, True, 20), (True, True, 20),
             (True, True, 40), (False, False, 20)]
    tags = [run_tag(*c) for c in cells]
    assert len(set(tags)) == len(tags)


def test_a_second_seed_never_overwrites_the_first():
    assert run_tag(False, True, 20, 2001) != run_tag(False, True, 20, 2000)
    assert run_tag(False, True, 20, 2001).endswith("_seed2001")
    assert run_tag(False, True, 20, 2000) == "rows"            # the baseline seed adds no suffix


def test_dwell_feature_gets_its_own_name_distinct_from_every_other_cell():
    cells = [(False, False, 40), (True, False, 20), (False, True, 20), (True, True, 20),
             (True, True, 40), (False, False, 20)]
    tags = [run_tag(*c) for c in cells]
    dwell_tag = run_tag(False, False, 20, dwell_feature=True)
    assert dwell_tag not in tags
    assert dwell_tag == "dwell"


def test_dwell_feature_composes_with_keep_rows_and_freq_summary_in_the_tag():
    assert run_tag(False, True, 20, dwell_feature=True) == "rows_dwell"
    assert run_tag(True, True, 40, dwell_feature=True) == "freq_rows_dwell_div40"
