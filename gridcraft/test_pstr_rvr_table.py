from run_benchmarl_gridcraft import pstr_rvr_rows
from wandb_schema import PSTR_RULES


def test_pstr_rvr_rows_keep_stable_rule_order():
    rows = pstr_rvr_rows({}, {"variant": "neural"})
    assert [row["PSTR id"] for row in rows] == [rule["id"] for rule in PSTR_RULES]


def test_pstr_rvr_rows_neural_keeps_available_values():
    first_rule = PSTR_RULES[0]["id"]
    rows = pstr_rvr_rows({first_rule: 0.25}, {"variant": "neural"})
    assert rows[0]["RVR"] == 0.25
    assert rows[1]["RVR"] == "n/a"


def test_pstr_rvr_rows_regularization_keeps_available_values():
    first_rule = PSTR_RULES[0]["id"]
    rows = pstr_rvr_rows({first_rule: 0.5}, {"variant": "regularization"})
    assert rows[0]["RVR"] == 0.5
    assert rows[1]["RVR"] == "n/a"


def test_pstr_rvr_rows_projection_is_all_na():
    first_rule = PSTR_RULES[0]["id"]
    rows = pstr_rvr_rows({first_rule: 0.25}, {"variant": "projection"})
    assert all(row["RVR"] == "n/a" for row in rows)


def test_pstr_rvr_rows_residual_is_all_na():
    first_rule = PSTR_RULES[0]["id"]
    rows = pstr_rvr_rows({first_rule: 0.25}, {"variant": "residual"})
    assert all(row["RVR"] == "n/a" for row in rows)
