import json
from dataclasses import replace
import pytest
import torch
from scipy import stats

from ns_mawm.schema import BlockSpec, Schema
from ns_mawm.rules import Assignment, Context, Fact, Library, PSWM, RuleSpec, Tri, compile_rule
from ns_mawm.models import JOPM, observation_loss, WorldModel
from ns_mawm.diagnostics import Diagnostics, error_split
from ns_mawm.statistics import compare, holm, wording
from ns_mawm.artifacts import export_anonymous
from ns_mawm.data import Dataset, Episode


@pytest.fixture
def schema():
    return Schema([BlockSpec("x", 0, "scalar", None, (0, 1)),
                   BlockSpec("c", 0, "categorical", ("a", "b", "c"), (1, 4))], (4, 1))


def rule(id="r", value="a", guard=Tri.TRUE, scope="individual"):
    return RuleSpec(id, 0, 0, scope, frozenset({"x", "c"}), frozenset({"c"}),
                    lambda c: guard, lambda c: [Assignment("c", value, id, 0)])


def context():
    return Context({"x": 0., "c": "a"}, (0,))


def test_t01_unknown(schema):
    out = PSWM(Library([rule(guard=Tri.UNKNOWN)], schema))(context())
    assert not out.mask.any() and not out.proposals


def test_t02_conflict(schema):
    out = PSWM(Library([rule(), rule("s", "b")], schema))(context())
    assert not out.mask.any() and len(out.conflicts[0]["proposals"]) == 2
    assert out.counters[("r", 0)]["categorical"]["conflicted"] == 1


def test_t03_order(schema):
    lib = Library([rule("z"), rule("a")], schema)
    assert lib.validate([context()])["valid"]
    assert PSWM(lib)(context()).provenance["c"] == [("a", 0), ("z", 0)]


def test_t04_semantic_blocks(schema):
    engine = PSWM(Library([rule()], schema))
    for values in ([0., 0., 1., 0.], [0., -.2, .8, .4]):
        d = Diagnostics(engine)
        d.add(context(), torch.tensor(values)[:, None], schema.encode(context().observation))
        assert d.report()["rules"][0]["rvr"] == 1
        assert d.report()["rules"][0]["support"] == 1


@pytest.mark.parametrize("code", ["import os", "def guard(c):\n return open('x','w')", "def guard(c):\n c.observation['x']=2", "def guard(c):\n return c.__class__", "def guard(c):\n return eval('1')"])
def test_t05_sandbox(code):
    with pytest.raises(ValueError):
        compile_rule({"code": code})


def test_t06_structural(schema):
    lib = Library([rule()], schema)
    for strategy in ("projection", "residual"):
        d = Diagnostics(PSWM(lib), enforced_hash=lib.hash, strategy=strategy)
        y = schema.encode(context().observation)
        d.add(context(), y, y)
        assert d.report()["rules"][0]["structural"]


def test_t07_zero_support(schema):
    d = Diagnostics(PSWM(Library([rule(guard=Tri.FALSE)], schema)))
    y = schema.encode(context().observation)
    d.add(context(), y, y)
    assert d.report()["rules"][0]["rvr"] is None


def test_t08_assigned_zero(schema):
    net = JOPM(schema, 2, enc=(4,), lstm=4, g=(4,), dec=(4,), residual=True)
    obs = torch.zeros(1, 4, 1)
    a = net.inputs(obs, (torch.ones_like(obs), obs))
    b = net.inputs(obs, (torch.zeros_like(obs), obs))
    assert not torch.equal(a, b)


def test_t09_t10_losses(schema):
    logits = torch.randn(1, 4, 1, requires_grad=True)
    target = schema.encode({"x": 0., "c": "b"})[None]
    mask = torch.ones_like(target)
    baseline = observation_loss(schema, logits, target, (mask, target), "none")
    zero = observation_loss(schema, logits, target, (mask, target), "regularization", 0)
    assert torch.equal(baseline, zero)
    reg = observation_loss(schema, logits, target, (mask, target), "regularization", .3)
    weight = observation_loss(schema, logits, target, (mask, target), "feature_weighting", .3)
    assert torch.equal(reg, weight)
    loss = observation_loss(schema, logits, target, (mask, target), "residual")
    assert loss == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_t11_memory(schema):
    fact = Fact("cached", "a", "local", 25, 0, "r", True)
    c = replace(context(), memory={"cached": fact})
    out = PSWM(Library([], schema))(c)
    assert "cached" not in out.memory
    with pytest.raises(TypeError):
        c.observation["x"] = 3


def dataset():
    episodes = [Episode(str(i), torch.zeros(3, 1, 1), torch.zeros(2, 1, dtype=torch.long),
                        torch.zeros(2), torch.tensor([False, True]), torch.zeros(2, dtype=torch.bool), i) for i in range(10)]
    return Dataset.split(episodes)


def test_t12_split_lock():
    d = dataset()
    with pytest.raises(PermissionError):
        d.get("test", purpose="prompt")
    with pytest.raises(PermissionError):
        d.get("test", purpose="evaluate")
    d.freeze("lib", {})
    assert d.get("test", purpose="evaluate")
    with pytest.raises(PermissionError):
        d.freeze("other", {})


def test_t14_statistics():
    a, b = [1., 2., 3., 4., 5.], [2., 4., 2., 7., 8.]
    result = compare(a, b, resamples=100)
    assert result["p_primary"] == pytest.approx(stats.wilcoxon(a, b).pvalue, abs=1e-9)
    assert result["p_secondary"] == pytest.approx(stats.ttest_rel(a, b).pvalue, abs=1e-9)
    assert holm([.04, .01, .03]) == pytest.approx([.06, .03, .06])


def test_t15_wording():
    assert wording(-1, .2) == "lower mean, not significant"


def test_t16_export(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "raw_runs.csv").write_text("author,Example Author\n")
    with pytest.raises(ValueError, match="raw_runs.csv"):
        export_anonymous(source, tmp_path / "anon.zip", deny=["Example Author"], replacements={}, benchmark_commits={"benchmark": "a"*40})


def test_t17_split_error(schema):
    target = schema.encode({"x": 0., "c": "a"})
    prediction = schema.encode({"x": 2., "c": "b"})
    mask = torch.tensor([[1.], [0.], [0.], [0.]])
    result = error_split(schema, prediction, target, mask, {"x"})
    assert result["covered_contribution"] + result["uncovered_contribution"] == result["mse"]
    assert result["mse"] == 2.5
    assert error_split(schema, prediction, target, 1-mask, {"x"})["fixed_error"] == result["fixed_error"]


def test_t18_filter(schema):
    lib = Library([rule(), rule("joint", scope="joint")], schema)
    filtered = lib.filtered(scope=["individual"])
    assert lib.hash != filtered.hash
    assert {a.rule_id for a in PSWM(filtered)(context()).proposals} == {"r"}


def test_schema_rejects_invalid():
    with pytest.raises(ValueError):
        Schema([], (1, 1))


def test_dag_rejected(schema):
    with pytest.raises(ValueError, match="Cyclic"):
        Library([rule()], schema, edges=[(0, 1), (1, 0)])


def test_conflict_not_structural(schema):
    lib = Library([rule(), rule("s", "b")], schema)
    diag = Diagnostics(PSWM(lib), enforced_hash=lib.hash, strategy="projection")
    y = schema.encode(context().observation)
    diag.add(context(), y, y, split="rev", evidence_id="e")
    assert all(not r["structural"] for r in diag.report()["rules"])
