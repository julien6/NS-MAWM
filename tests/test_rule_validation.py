from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from ns_mawm.schema import Schema, BlockSpec
from ns_mawm.rules import Assignment, Context, Fact, Library, PSWM, RuleSpec, Tri, compile_rule, time_limit, freeze
from ns_mawm.diagnostics import Diagnostics, dashboard, union_blocks


@pytest.fixture
def schema():
    return Schema([BlockSpec("x", 0, "scalar", None, (0,1), comp_tol=.1),
                   BlockSpec("cat[0]", 0, "categorical", ("a","b"), (1,3))], (3,1))


def compiled(code=None, **kwargs):
    record = {"id":"r", "version":0, "stage":0, "scope":"individual", "reads":["x"], "writes":["x"],
              "code": code or "def guard(c):\n return Tri.TRUE\ndef effect(c):\n return [Assignment('x', c.observation['x'] + 1, 'r', 0)]"}
    record.update(kwargs)
    return compile_rule(record)


def c():
    return Context({"x":0., "cat[0]":"a"}, (0,))


def test_compiled_execution_and_fallback(schema):
    engine = PSWM(Library([compiled()], schema))
    out = engine(c())
    assert out.target[0] == 1
    assert engine.predict(c())[0] == 1
    assert engine.predict(c(), torch.ones(3,1))[1] == 1
    assert engine.batch([c(), c()])[1].mask[0] == 1
    assert union_blocks([out]) == {"x"}


@pytest.mark.parametrize("changes", [{"scope":"bad"}, {"version":-1}, {"reads":frozenset({"missing"})}, {"writes":frozenset()}, {"writes":frozenset({"missing"})}])
def test_invalid_metadata(schema, changes):
    with pytest.raises(ValueError):
        Library([replace(compiled(), **changes)], schema)


def test_duplicate_policy_context(schema):
    with pytest.raises(ValueError): Library([compiled(),compiled()], schema)
    with pytest.raises(ValueError): Library([], schema, policy="bad")
    with pytest.raises(ValueError): Context({}, (), origin="future")
    with pytest.raises(TypeError): freeze(torch.ones(1))
    with pytest.raises(ValueError): Context({}, (), history=({},)*51)


@pytest.mark.parametrize("body", ["return [Assignment('x', 2, 'wrong', 0)]", "return [Assignment('x', 2, 'r', 0), Assignment('x', 3, 'r', 0)]", "return [Assignment('cat[0]', 'a', 'r', 0)]", "return [Assignment('x', float('nan'), 'r', 0)]"])
def test_invalid_assignments(schema, body):
    rule = compiled("def guard(c):\n return Tri.TRUE\ndef effect(c):\n " + body)
    with pytest.raises((ValueError, TypeError)):
        PSWM(Library([rule], schema))(c())


def test_invalid_guard(schema):
    r=replace(compiled(), guard=lambda c: True)
    with pytest.raises(TypeError): PSWM(Library([r],schema))(c())


def test_stage_snapshot_and_fact_conflicts(schema):
    def facts(context): return [Fact("memo", 1., "world", 0, context.step, "r", True)]
    first = replace(compiled(), derives=facts)
    second = replace(compiled(), id="s", stage=1, reads=frozenset({"memory:memo"}),
        effect=lambda c: [Assignment("x", c.memory["memo"].value, "s",0)])
    lib=Library([first,second], schema, edges=[(0,1)])
    out=PSWM(lib)(c())
    assert out.derived_facts[0].value == 1 and out.mask[0] == 1
    assert lib.validate([c()])["valid"]
    contradiction = replace(first, id="other", effect=lambda c:[], derives=lambda c:[Fact("memo",2.,"world",0,c.step,"other",True)])
    assert "memo" not in PSWM(Library([first,contradiction],schema))(c()).memory
    invalid = replace(first, derives=lambda c:[Fact("memo",1.,"world",0,0,"wrong",True)])
    with pytest.raises(ValueError): PSWM(Library([invalid],schema))(c())


def test_scalar_canonical_and_reject_policy(schema):
    first=compiled()
    agree=replace(first,id="z",effect=lambda c:[Assignment("x",1.05,"z",0)])
    out=PSWM(Library([first,agree],schema))(c())
    assert out.target[0] == 1
    conflict=replace(agree,effect=lambda c:[Assignment("x",2.,"z",0)])
    with pytest.raises(ValueError): PSWM(Library([first,conflict],schema,policy="reject_library"))(c())
    assert not Library([first,conflict],schema,policy="reject_library").validate([c()])["valid"]


def test_case_validation(schema):
    case={"context":{"observation":{"x":0.,"cat[0]":"a"},"joint_action":[0]},"guard":"TRUE","assignments":{"x":1.}}
    r=replace(compiled(),test_cases=(case,))
    lib=Library([r],schema)
    assert lib.validate()["valid"]
    assert not lib.validate(require_cases=True)["valid"]
    wrong={**case,"guard":"FALSE"}
    assert not Library([replace(r,test_cases=(wrong,))],schema).validate()["valid"]
    wrong={**case,"assignments":{"x":2.}}
    assert not Library([replace(r,test_cases=(wrong,))],schema).validate()["valid"]
    assert lib.swapped(replace(r,version=1)).parent == lib.hash


def test_filter_tags_and_read_restrictions(schema):
    r=replace(compiled(), tags=("physics",))
    lib=Library([r],schema).filtered(tags=["other"])
    assert not PSWM(lib)(c()).proposals
    r=replace(r,reads=frozenset(),effect=lambda c:[Assignment("x", c.observation["x"],"r",0)])
    with pytest.raises(KeyError): PSWM(Library([r],schema))(c())


def test_timeout_thread_and_allocation(schema):
    import time
    with pytest.raises(TimeoutError):
        with time_limit(.001): time.sleep(.01)
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError): pool.submit(lambda: PSWM(Library([compiled()],schema))(c())).result()
    r=compiled("def guard(c):\n return Tri.TRUE\ndef effect(c):\n a = [0]\n b = a * 1000000000\n return []")
    with pytest.raises(ValueError): PSWM(Library([r],schema))(c())


@pytest.mark.parametrize("code", ["def guard(c):\n return 2 ** 10", "def guard(c):\n return _private", "x = 1", "def guard(c):\n return Tri.TRUE", "def guard(c):\n x = '"+"a"*10001+"'", "def guard(c):\n x += 1"])
def test_invalid_subset(code):
    with pytest.raises((ValueError, KeyError)):
        compiled(code)


def test_dashboard_and_interpretation(schema,tmp_path):
    lib=Library([compiled()],schema)
    diagnostic=Diagnostics(PSWM(lib),strategy="projection",enforced_hash=lib.hash)
    obs=schema.encode(c().observation)
    prediction=obs.clone();prediction[0]=1
    diagnostic.add(c(),prediction,obs,pre=obs,split="rev",evidence_id="rev:0")
    report=diagnostic.report()
    assert report["rules"][0]["rdd"] == 1
    assert report["rules"][0]["rvr_pre"] == 1
    dashboard({"model":report},tmp_path / "report.html")
    assert "Evidence" in (tmp_path / "report.html").read_text()
