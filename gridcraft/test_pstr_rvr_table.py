from run_benchmarl_gridcraft import pstr_rvr_rows
from wandb_schema import PSTR_RULES


def test_pstr_rvr_rows_keep_stable_rule_order():
    rows = pstr_rvr_rows({}, {}, {})
    assert [row["PSTR id"] for row in rows] == [rule["id"] for rule in PSTR_RULES]


def test_pstr_rvr_rows_neural_keeps_available_values():
    first_rule = PSTR_RULES[0]["id"]
    rows = pstr_rvr_rows({first_rule: 0.25}, {}, {first_rule: 49})
    assert rows[0]["RVR pre"] == 0.25
    assert rows[0]["RVR post"] == "n/a"
    assert rows[0]["Determinable count"] == 49
    assert rows[1]["RVR pre"] == "n/a"


def test_pstr_rvr_rows_regularization_keeps_available_values():
    first_rule = PSTR_RULES[0]["id"]
    rows = pstr_rvr_rows({first_rule: 0.5}, {}, {})
    assert rows[0]["RVR pre"] == 0.5
    assert rows[0]["RVR post"] == "n/a"
    assert rows[1]["RVR pre"] == "n/a"


def test_pstr_rvr_rows_projection_can_show_pre_and_post_separately():
    first_rule = PSTR_RULES[0]["id"]
    rows = pstr_rvr_rows({first_rule: 0.25}, {first_rule: 0.0}, {first_rule: 49})
    assert rows[0]["RVR pre"] == 0.25
    assert rows[0]["RVR post"] == 0.0
    assert rows[0]["Determinable count"] == 49


def test_pstr_rvr_rows_missing_values_are_na():
    first_rule = PSTR_RULES[0]["id"]
    rows = pstr_rvr_rows({}, {}, {})
    assert rows[0]["RVR pre"] == "n/a"
    assert rows[0]["RVR post"] == "n/a"
    assert rows[0]["Determinable count"] == "n/a"
