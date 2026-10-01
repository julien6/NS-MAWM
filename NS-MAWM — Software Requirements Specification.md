# NS-MAWM — Software Requirements Specification

Oct 1, 2026 · @Julien Soule

## 1. Introduction

This SRS specifies the final NS-MAWM framework to be re-implemented from scratch: the current implementation described in the AAMAS 2027 draft, plus every extension needed to answer the ten review concerns. The central contribution is rule-level diagnostics and diagnostic rule refinement; the three integration strategies are the testbed on which those diagnostics are exercised.

### 1.1 Purpose

The document is the single reference for engineers and researchers who build, test and run NS-MAWM. It is written so that a team with no access to the original code can produce a functionally equivalent system and reproduce every experiment in the paper, including the new ones.

### 1.2 Scope

NS-MAWM is a Python library plus an experiment harness. It covers:

- a partial symbolic world model (PSWM) built from executable, versioned partial symbolic transition rules (PSTRs);
- neural joint-observation world models and three integration strategies (Projection, Residual, Regularization);
- per-rule diagnostics (RVR, RDD, support, coverage, conflicts, covered/uncovered error);
- an LLM workflow that generates and revises PSTRs from diagnostic evidence;
- environment adapters and rule libraries for Gridcraft, Overcooked, Predator–Prey and SMACv2;
- integration into a model-based MARL learner;
- baselines, ablations, statistics and reproducibility tooling.

Out of scope: stochastic symbolic effects, inequality-only constraints, decentralized symbolic prediction and human-subject studies of interpretability. These are listed as future extensions in Section 13.

### 1.3 Definitions

| Term | Definition |
| --- | --- |
| Joint observation | n\_f × n matrix; one column per agent, n\_f scalar entries after encoding |
| Semantic block | Unit of logical evaluation: one scalar attribute or one complete categorical block (e.g. one terrain cell) |
| Context x\_t | (h^j\_{t-1}, ω^j\_t, a^j\_t): explicit joint history, current joint observation, current joint action |
| PSTR | Rule (id, version, guard, effect) returning predicted values Y\_r and a mask M\_r; abstains when its guard is false or unknown |
| Joint PSTR | PSTR that reads more than one agent's observation or action |
| PSWM | Rule library + symbolic memory + execution schedule + conflict policy; returns (Y, M, provenance, conflicts) |
| Coverage | Fraction of semantic blocks with an accepted symbolic assignment in a context |
| Support S\_r | Number of (context, block) pairs that rule r assigns on a dataset |
| RVR\_r | Fraction of r's assigned blocks where a predictor disagrees with r |
| RDD\_r | Fraction of r's assigned blocks where recorded data disagrees with r |
| Covered / uncovered error | Prediction error restricted to blocks with / without an accepted assignment |
| JOPM | Joint-observation prediction model: the centralized neural backbone |
| MSE\_H | Normalized block error over H-step open-loop rollouts (H = 25 by default) |

### 1.4 Conventions

- **shall** = mandatory; **should** = expected unless justified; **may** = optional.
- Requirement IDs use a prefix per area: FR-PSWM, FR-NN, FR-INT, FR-DIAG, FR-LLM, FR-ENV, FR-MB, FR-BASE, FR-EXP, FR-STAT, FR-REPRO, NFR.
- Priority: **M** (must, needed for the submission), **S** (should), **C** (could).
- Origin: **\[BASE\]** = present in the current implementation; **\[NEW\]** = extension added to address a review concern, with the concern number (e.g. \[NEW C6\]).

### 1.5 References

- NS-MAWM draft, "Rule-Level Diagnostics for Neuro-Symbolic Multi-Agent World Models", AAMAS 2027 submission.
- Review concerns C1–C10 (Section 15 maps each to requirements).
- Benchmarks: Gridcraft, Overcooked-AI, MPE Predator–Prey, SMACv2. Learner: BenchMARL MASAC/MAMBPO.
- Related systems to compare against or cite: MAMBA (Egorov & Shpilman, AAMAS 2022), DreamerV3-style latent models, APHYNITY (Yin et al., ICLR 2021), WorldCoder, code world models (Dainese et al., NeurIPS 2024).

## 2. Overall description

NS-MAWM is a modular PyTorch library in which a symbolic rule engine and a neural joint-observation predictor share one observation schema, and every prediction can be traced back to the rule versions that produced or constrained it.

### 2.1 Product perspective

The system sits between multi-agent environments and a model-based MARL learner. It consumes recorded or live joint transitions through environment adapters. It produces trained world models, frozen rule libraries, per-rule diagnostic reports, and imagined transitions for policy learning. It is self-contained except for the environments, the LLM server and the BenchMARL learner.

&#91;embedded content: NS-MAWM architecture · 8 components, diagnostics-driven rule loop\]

The diagnostics engine scores every model against each rule; its report drives LLM revisions, which return to the PSWM as new rule versions.

### 2.2 Components

| Component | Responsibility | Origin |
| --- | --- | --- |
| Schema registry | Declares semantic blocks, encodings, units, tolerances per environment | \[BASE\] |
| Environment adapters | Map native observations/actions to the schema; record episodes | \[BASE\], extended to all four envs \[NEW C3\] |
| Rule engine (PSWM) | Executes PSTRs, symbolic memory, composition, conflicts, provenance | \[BASE\] |
| Neural backbones | JOPM-LSTM; external baselines (MAMBA-style, Dreamer-style) | JOPM \[BASE\]; external \[NEW C4\] |
| Integration layer | Projection, Residual, Regularization; feature-weighting control | \[BASE\] |
| Rollout engine | Open-loop and policy-driven imagination with neural and symbolic memory | \[BASE\] |
| Diagnostics engine | RVR, RDD, support, coverage, conflicts, covered/uncovered error | \[BASE\]; split + conflict rates \[NEW C6\] |
| LLM workflow | P0–P3 prompts, validation, candidate archive, selection, cost ledger | \[BASE\]; cost + leakage logging \[NEW C8, C9\] |
| MBRL bridge | Reward/termination heads, imagined replay for MASAC | \[BASE\] |
| Experiment orchestrator | Configs, seeds, splits, sweeps, ablation grid | \[BASE\]; new grid \[NEW C4, C7, C8\] |
| Statistics and reporting | Paired tests, CIs, multiple-comparison correction, tables, figures | \[NEW C5\] |
| Artifact store | Run manifests, hashes, prompts/responses, anonymized export | \[NEW C8, C9\] |

### 2.3 User classes

| User | Main tasks |
| --- | --- |
| Method researcher | Adds strategies, backbones or diagnostics |
| Rule author | Writes or reviews PSTRs for a new environment |
| Experimenter | Runs the experiment grid and produces paper tables |
| Auditor / reviewer | Reruns a configuration from the artifact package and checks provenance |

### 2.4 Operating environment

- Linux, Python ≥ 3.10, PyTorch ≥ 2.2, CUDA GPU (one 24 GB+ GPU per run is sufficient for prediction experiments).
- LLM: Qwen3.6-35B-A3B in NVFP4, served through an OpenAI-compatible local endpoint; the model identifier is configurable.
- StarCraft II client for SMACv2; pinned commits for all benchmark repositories.

### 2.5 Design constraints

- Rules shall never read simulator state, hidden variables, or future observations.
- Centralized model access is a training privilege; deployed policies use local histories only.
- The neural predictor is deterministic; stochastic latent models are outside the core but external baselines may be stochastic.
- Test transitions shall never enter prompts, revision, selection or hyperparameter tuning.

### 2.6 Assumptions and dependencies

- Each environment exposes a structured (non-pixel) observation that can be mapped to semantic blocks.
- Action-resolution semantics (collision handling, order of simultaneous effects) are documented or can be stated by the rule author; the source of that knowledge is recorded (FR-LLM-12).
- Benchmark repositories remain available at their pinned commits.

## 3. Data model and core interfaces

All components exchange the typed objects below; rules address semantic block names, never raw tensor offsets.

### 3.1 Observation schema

- **FR-SCH-1 (M) \[BASE\]** Each environment shall declare a schema: an ordered list of `BlockSpec` covering every encoded entry of the joint observation exactly once.
- **FR-SCH-2 (M) \[BASE\]** A categorical block (e.g. one terrain cell) shall count as one semantic block regardless of its one-hot width.
- **FR-SCH-3 (M) \[BASE\]** Each scalar block shall declare its unit, training-set scaling (mean, std), evaluation tolerance ε\_b in native units, and a separate composition tolerance.
- **FR-SCH-4 (S) \[NEW C7\]** Each block shall declare an `owner` (agent index or `shared`) so diagnostics can be grouped by individual vs. cross-agent content.

### 3.2 Typed contracts

```python
class Tri(Enum): TRUE; FALSE; UNKNOWN

@dataclass(frozen=True)
class BlockSpec:
    name: str                 # e.g. "agent0.terrain[2,3]"
    owner: int | Literal["shared"]
    kind: Literal["scalar", "categorical"]
    categories: tuple[str, ...] | None
    slice: tuple[int, int]    # rows in the n_f x n tensor, column = owner
    unit: str | None
    scale: tuple[float, float] | None
    eval_tol: float | None    # epsilon_b, native units
    comp_tol: float | None    # composition tolerance (0 = exact)

@dataclass(frozen=True)
class Context:
    history: JointHistory     # bounded explicit window, real or imagined
    observation: JointObs     # omega^j_t (decoded, immutable)
    joint_action: JointAction # a^j_t
    prev_joint_action: JointAction | None
    memory: SymbolicMemory    # read-only view
    origin: Literal["real", "imagined"]

@dataclass(frozen=True)
class Assignment:
    block: str
    value: float | str        # category name for categorical blocks
    rule_id: str
    rule_version: int

@dataclass(frozen=True)
class RuleSpec:
    id: str                   # stable across versions, e.g. "terrain.shift"
    version: int
    stage: int                # position in the declared stage DAG
    scope: Literal["individual", "joint"]
    reads: frozenset[str]     # block names or memory keys
    writes: frozenset[str]    # block-name patterns
    guard: Callable[[Context], Tri]
    effect: Callable[[Context], list[Assignment]]
    derives: Callable[[Context], list[Fact]] | None
    assumptions: tuple[str, ...]
    source: Literal["handcrafted", "llm_initial", "llm_refined", "human_edited"]
    base_version: int | None
    code_hash: str
    test_cases: tuple[RuleTestCase, ...]

@dataclass
class PSWMOutput:
    target: Tensor            # Y_t, full observation shape, zero padding
    mask: Tensor              # M_t in {0,1}, full shape
    provenance: dict[str, list[tuple[str, int]]]
    conflicts: list[ConflictRecord]
    proposals: list[Assignment]     # unaggregated, for diagnostics
    derived_facts: list[Fact]
    memory: SymbolicMemory          # m_t after update
```

### 3.3 Interface rules

- **FR-IF-1 (M) \[BASE\]** A guard returning `UNKNOWN` shall be treated as abstention (empty mask).
- **FR-IF-2 (M) \[BASE\]** Effects shall not mutate the context, access files, network, simulator objects or future labels. The engine shall enforce this by passing frozen copies and executing rule code in a restricted namespace.
- **FR-IF-3 (M) \[BASE\]** Every assignment shall be validated for: known block name, finite value, legal category, complete categorical block, and membership in the rule's declared `writes`.
- **FR-IF-4 (M) \[BASE\]** A predicate that only bounds a value ("speed ≤ v\_max") is not a PSTR and shall be rejected at load time unless it assigns an exact value.
- **FR-IF-5 (M) \[BASE\]** Any change to guard, effect, reads or writes shall create a new version; identifiers stay stable.

### 3.4 Library manifest

A `LibraryVersion` records: library id, parent library id, the ordered list of (rule id, version, code hash), the stage DAG, the conflict policy, the creator (human, LLM call id, or both), and a content hash. Every trained model, diagnostic report and result row references exactly one library hash.

## 4. PSWM engine

The engine turns a context into an accepted symbolic prediction with full provenance, and its acceptance result shall not depend on rule order.

### 4.1 Execution

- **FR-PSWM-1 (M) \[BASE\]** Memory update m\_t = U(m\_{t−1}, ω^j\_t, a^j\_{t−1}) shall run before rule execution. At t = 0 the previous action is a null symbol. U incorporates the current observation, invalidates contradicted facts and keeps only facts whose persistence is justified.
- **FR-PSWM-2 (M) \[BASE\]** Each cached fact shall carry its coordinate frame, age in steps, source step and source rule. A configurable maximum age shall expire facts.
- **FR-PSWM-3 (M) \[BASE\]** Rules shall run in the order of a declared acyclic stage graph. Within a stage, all rules read the same immutable snapshot. Derived facts become visible only to later stages. A cycle shall be a load-time error.
- **FR-PSWM-4 (M) \[BASE\]** Memory updates and derived facts shall be logged separately from next-observation assignments and shall not receive RVR.
- **FR-PSWM-5 (M) \[BASE\]** The engine shall return `PSWMOutput` (Section 3.2), including unaggregated proposals.

### 4.2 Composition and conflicts

- **FR-PSWM-6 (M) \[BASE\]** Default policy is conservative block rejection: a block is accepted only if all applicable proposals agree (categorical: same category; scalar: within the composition tolerance). The canonical value is taken from the lowest rule id. Disagreeing blocks stay uncovered and are logged in `conflicts`.
- **FR-PSWM-7 (S) \[BASE\]** An alternative policy shall reject an entire library that produces any conflict on its validation cases. The active policy is stored in the library manifest.
- **FR-PSWM-8 (M) \[BASE\]** Partial categorical assignments shall be rejected; padding shall never be decoded as a category.
- **FR-PSWM-9 (M) \[NEW C6\]** The engine shall count, per rule and per block type: proposed blocks, accepted blocks, blocks rejected by conflict, and blocks abstained. These counts feed the conflict rate and the proposed-vs-accepted coverage reported in Section 6.

### 4.3 Library loading and validation

- **FR-PSWM-10 (M) \[BASE\]** Loading a library shall run: parse; type checks; schema checks on reads/writes; DAG check; sandbox check; and each rule's test cases (applicable, inapplicable, unknown-premise). Failures produce a structured report usable by prompt P2.
- **FR-PSWM-11 (M) \[BASE\]** An order-invariance test shall execute each library under at least three random rule orders within stages and assert identical `(Y, M)`.

### 4.4 Configuration hooks for experiments

- **FR-PSWM-12 (M) \[NEW C7\]** A rule filter shall enable or disable rules by id, by scope (individual / joint) or by tag, without editing rule code. The filter is part of the run configuration and the library hash.
- **FR-PSWM-13 (M) \[NEW C4\]** A standalone symbolic predictor mode shall return a full observation: accepted symbolic values on covered blocks and a declared fallback (copy of the current observation block, or the training-set mode/mean) on uncovered blocks. It implements the PSWM-only baseline.
- **FR-PSWM-14 (S) \[BASE\]** The engine shall support batched execution over many contexts. Rule code may be per-context Python; a vectorized path is optional (see NFR-PERF).

### 4.5 Reference algorithm

```text
Input: m_{t-1}, ω^j_t, a^j_{t-1}, a^j_t, library R, filter F
1  m_t ← U(m_{t-1}, ω^j_t, a^j_{t-1})
2  proposals ← [], conflicts ← [], facts ← []
3  for stage in topological_order(R):
4      snapshot ← freeze(ω^j_t, a^j_t, m_t, facts)
5      for r in stage if F(r):
6          g ← r.guard(snapshot)
7          if g == TRUE:
8              proposals += validate(r.effect(snapshot))
9              new_facts += r.derives(snapshot)
10     facts += compatible(new_facts)
11 group proposals by block
12 accept agreeing blocks; reject conflicting blocks; log all
13 build Y_t, M_t, provenance P_t, conflicts C_t, counters
14 return m_t, PSWMOutput
```

## 5. Neural world models and integration strategies

The three strategies share one backbone interface and one block-loss definition, so differences between them come only from where symbolic predictions enter.

### 5.1 Backbone interface

- **FR-NN-1 (M) \[NEW C4\]** Every backbone shall implement `init_state(prefix)`, `step(state, obs, joint_action, extra=None) → (block_outputs, state)` and expose per-block outputs (logits for categorical blocks, values for scalars). `extra` carries (M\_t, Y\_t) for Residual. Any backbone meeting this interface can be combined with all three strategies.
- **FR-NN-2 (M) \[BASE\]** Reference backbone (JOPM): z\_t = Enc(flatten ω^j\_t); h̃\_t = LSTM(h̃\_{t−1}, \[z\_t, flatten a^j\_t\]); ẑ\_{t+1} = g(h̃\_t); outputs = Dec(ẑ\_{t+1}). Enc, g and Dec are MLPs.
- **FR-NN-3 (M) \[NEW C8\]** All sizes shall live in configuration and be printed in the results appendix. Defaults: Enc 2×256, LSTM 256, g 2×256, Dec 2×256, ReLU, Adam lr 3e−4, batch 32 sequences × 50 steps, burn-in 10 steps.
- **FR-NN-4 (M) \[NEW C8\]** "Update" shall mean one optimizer step on one minibatch. Epochs, dataset passes and the number of updates per run shall be reported separately; checkpoints are indexed by update.
- **FR-NN-5 (M) \[BASE\]** Reward and termination heads shall condition on the recurrent state, use identical architectures and losses across all variants, and be trained only when policy learning is enabled.

### 5.2 Block losses

- **FR-NN-6 (M) \[BASE\]** Scalar block loss: squared error in training-set-scaled units. Categorical block loss: cross-entropy (training); Brier-style error ‖p\_b − onehot(y\_b)‖²/2 (evaluation).
- **FR-NN-7 (M) \[BASE\]** The data loss averages block losses over all blocks; every semantic block weighs the same regardless of encoding width.

### 5.3 Projection

```latex
F_{\mathrm{proj}}(x_t) = M_t \odot Y_t + (1 - M_t) \odot N_\theta(x_t)
```

- **FR-INT-1 (M) \[BASE\]** The neural model is trained on the data loss only. At inference, accepted symbolic blocks overwrite neural outputs.
- **FR-INT-2 (M) \[BASE\]** The pre-enforcement output N\_θ(x\_t) shall remain accessible for diagnostics (RVR\_pre).
- **FR-INT-3 (M) \[BASE\]** Corrected observations feed the next imagined step; the neural hidden state is not reset.

### 5.4 Residual

```latex
Z_t = G_\theta(\tilde h_{t-1}, \omega^j_t, a^j_t, M_t, Y_t), \quad F_{\mathrm{res}}(x_t) = M_t \odot Y_t + (1 - M_t) \odot Z_t
```

- **FR-INT-4 (M) \[BASE\]** G\_θ receives the full observation, the mask and the zero-padded symbolic target, so that an assigned zero is distinguishable from abstention.
- **FR-INT-5 (M) \[BASE\]** The observation loss averages over uncovered blocks only; it is zero when every block is covered.
- **FR-INT-6 (M) \[BASE\]** Output slots are dense (full shape). A fixed reduced-output implementation, if added, shall be a separately named variant.

### 5.5 Regularization

```latex
\mathcal{L}_{\mathrm{reg}} = \mathcal{L}_{\mathrm{data}} + \lambda \, \frac{\sum_{b \in \mathcal{B}^d_t} \ell_b(N_\theta(x_t)_b, Y_{t,b})}{\max(1, |\mathcal{B}^d_t|)}, \quad F_{\mathrm{reg}}(x_t) = N_\theta(x_t)
```

- **FR-INT-7 (M) \[BASE\]** Each accepted block contributes once to the symbolic loss, even when several rules assign it. Symbolic targets are constants with respect to θ.
- **FR-INT-8 (M) \[BASE\]** At inference, Regularization runs the PSWM only when diagnostics are requested.
- **FR-INT-9 (M) \[NEW C8\]** λ shall be swept over {0, 0.1, 0.3, 1, 3} on D\_select. The selected value and the full sweep curve (MSE\_25 and RVR per λ) shall be reported.

### 5.6 Feature-weighting control

- **FR-INT-10 (M) \[BASE\]** A control model shall upweight covered blocks in the data loss by the same effective weight as Regularization (1 + λ times the per-block share), using observed targets instead of symbolic ones. It isolates the effect of reweighting from the effect of rule content.

### 5.7 Rollout engine

- **FR-ROLL-1 (M) \[BASE\]** A rollout initializes neural state and symbolic memory from the same real prefix, then uses only imagined observations and replayed (evaluation) or sampled (policy learning) joint actions.
- **FR-ROLL-2 (M) \[BASE\]** Predictions shall be decoded and re-encoded before being fed back; accepted assignments are preserved where the strategy enforces them.
- **FR-ROLL-3 (M) \[BASE\]** Rules shall never read hidden state or real future observations to repair an imagined trajectory. Imagined contexts carry `origin = "imagined"`.
- **FR-ROLL-4 (M) \[BASE\]** Rollouts stop at the horizon or at predicted termination.

## 6. Per-rule diagnostics

Diagnostics are the primary contribution: for every rule version they report how often a model disagrees with it, how often data disagrees with it, how much evidence supports each rate, and how much prediction error sits inside versus outside rule coverage.

### 6.1 Core rates

```latex
v_b(u,y) = \begin{cases} \mathbb{1}[\mathrm{decode}(u) \neq \mathrm{decode}(y)] & b \text{ categorical} \\ \mathbb{1}[|u - y| > \epsilon_b] & b \text{ scalar} \end{cases} \qquad S_r(D) = \sum_{x \in D} \sum_{b} a_{r,b}(x)
```

```latex
\mathrm{RVR}_r(F;D) = \frac{\sum_{x,b} a_{r,b}(x)\, v_b(F(x)_b, Y_r(x)_b)}{S_r(D)}, \qquad \mathrm{RDD}_r(D) = \frac{\sum_{x,b} a_{r,b}(x)\, v_b(\omega^j_+(x)_b, Y_r(x)_b)}{S_r(D)}
```

- **FR-DIAG-1 (M) \[BASE\]** RVR and RDD shall be computed per rule version from integer counts (violations, support). Zero support reports N/A, never 0.
- **FR-DIAG-2 (M) \[BASE\]** RVR uses each rule's own unaggregated proposals, so overlapping rules are diagnosed separately.
- **FR-DIAG-3 (M) \[BASE\]** A fixed diagnostic library R\_eval shall be applied to every predictor, including those trained without rules.
- **FR-DIAG-4 (M) \[BASE\]** For Projection, both RVR\_pre (on N\_θ) and RVR\_out (on F\_proj) shall be reported. Residual reports RVR\_out only. When enforcement and diagnostic libraries coincide, RVR\_out = 0 shall be labelled "structural" in every output table.
- **FR-DIAG-5 (S) \[BASE\]** An application-level variant (an application is violated if any assigned block disagrees) and macro, micro and union-mask aggregates shall be available, each named distinctly.

### 6.2 Coverage, conflicts and error split

- **FR-DIAG-6 (M) \[NEW C6\]** For every evaluation set and library, the system shall report: proposed coverage, accepted coverage, fraction rejected by conflict, and per-rule conflict involvement.
- **FR-DIAG-7 (M) \[NEW C6\]** MSE\_H shall be decomposed into covered-block error and uncovered-block error, using the accepted mask of a fixed reference library on the real context at each step. The decomposition shall be reported for every strategy, every rule source and every refinement version.
- **FR-DIAG-8 (M) \[NEW C6\]** To separate "better predictions" from "abstaining on hard blocks", the system shall also report MSE on a fixed block set: the union of blocks covered by any compared library version, evaluated identically for all versions.

### 6.3 Evaluation contexts

- **FR-DIAG-9 (M) \[BASE\]** The principal cross-model comparison uses identical real contexts, recurrent warm-up, rule versions and masks.
- **FR-DIAG-10 (S) \[BASE\]** A second mode evaluates rules on each model's own imagined history and reports coverage per rollout step, flagging premises taken from imagined contexts.
- **FR-DIAG-11 (M) \[BASE\]** Rates shall be computed per run before aggregation across runs.

### 6.4 Grouping for the multi-agent analysis

- **FR-DIAG-12 (M) \[NEW C7\]** Every rate and coverage figure shall be groupable by rule scope (individual vs joint) and by block owner (own agent vs other agent vs shared), so the contribution of cross-agent knowledge is visible.

### 6.5 Interpretation aids

The report shall attach the following candidate reading to each supported rule, using thresholds declared before test access:

| RVR | RDD | Candidate interpretation |
| --- | --- | --- |
| Low | Low | Model and rule agree on observed cases |
| High | Low | Model error, optimization difficulty or tolerance mismatch |
| Low | High | Model may have absorbed an unsuitable rule |
| High | High | Inspect both model error and rule premises/effects |

- **FR-DIAG-13 (M) \[BASE\]** Each diagnostic report shall attach up to k (default 5) counterexamples per flagged rule from D\_rev: rule version, context origin, guard result, assigned blocks, symbolic target, neural prediction, recorded observation.
- **FR-DIAG-14 (S) \[NEW C1\]** A rule-level dashboard (HTML export) shall show, per rule: support, RVR per model, RDD, coverage, conflicts and counterexamples, so the diagnostic contribution can be shown as a figure in the paper.

## 7. LLM rule generation and diagnostic refinement

The LLM proposes executable rule code; all validation, training, measurement and selection happen outside it, on data splits it never sees beyond the revision set.

&#91;embedded content: Rule refinement loop · P1–P3 with external validation and selection\]

Executable failures loop through P2; diagnostic evidence loops through P3; D\_test is opened once, after the budget ends and the library is frozen.

### 7.1 Data separation

- **FR-LLM-1 (M) \[BASE\]** Complete episodes shall be split before any training into D\_train, D\_rev, D\_select and D\_test (default 60/15/10/15 %). The split manifest (episode ids, seeds, hash) is stored and referenced by every run.
- **FR-LLM-2 (M) \[BASE\]** D\_test shall be inaccessible to prompts, revision, selection and hyperparameter tuning. The orchestrator enforces this with a read lock released only after the library and all hyperparameters are frozen.

### 7.2 Prompts

| Prompt | Inputs | Required output |
| --- | --- | --- |
| P0 contract (prepended to all) | Allowed context and memory; block schema; rule API; response schema | Abstain on unknown premises; complete categorical blocks; preserve ids and versions; no simulator or future access |
| P1 generation | P0 + domain description, action semantics, recipes, interaction-resolution rules | Library of individual and joint PSTRs with guards, effects, reads/writes, assumptions, test cases |
| P2 repair | P0 + exact candidate + validation report | Minimal patch with base and new version, affected ids, change summary |
| P3 revision | P0 + library + per-rule report (RVR, RDD, support, conflicts, coverage) + counterexamples | Diagnosis with evidence ids; versioned patch or explicit no-change |

- **FR-LLM-3 (M) \[BASE\]** Every response shall follow one JSON schema: base\_library\_version, candidate\_version, rule\_id, rule\_version, operation (add / revise / remove / retain), read\_blocks, write\_blocks, dependencies, guard\_code, effect\_code, assumptions, evidence\_ids, change\_summary, test\_cases. Invalid JSON triggers a schema-repair retry that counts against the budget.
- **FR-LLM-4 (M) \[BASE\]** P3 shall instruct the model not to lower RVR by copying neural outputs or by removing coverage without evidence; it may answer "no change".
- **FR-LLM-5 (M) \[BASE\]** Counterexamples sent to P3 come from D\_rev only and state whether premises came from a real or imagined context.

### 7.3 Candidate evaluation and selection

- **FR-LLM-6 (M) \[BASE\]** Each executable candidate trains a Regularization model on D\_train from the same seed-specific initialization and the same update budget as other candidates. Warm-starting, if used, is a separate variant with an equal-total-updates control.
- **FR-LLM-7 (M) \[BASE\]** A candidate is admissible if it passes validation, its conflict rate ≤ τ\_c, every enforced rule has RDD ≤ τ\_rdd with support ≥ S\_min, and accepted coverage on D\_select ≥ the coverage floor. Defaults: τ\_c = 5 %, τ\_rdd = 10 %, S\_min = 50, floor = 50 % of the handcrafted library's coverage. Rules below S\_min are marked unvalidated and excluded from enforcement.
- **FR-LLM-8 (M) \[BASE\]** Selection minimizes MSE\_25 on D\_select among admissible candidates; ties go to fewer rules, then lower generation cost.
- **FR-LLM-9 (M) \[BASE\]** Budget: K revision rounds producing at most 4 candidates (default), plus a token budget. If no candidate is admissible, the system reports failure and falls back to the purely neural model.
- **FR-LLM-10 (M) \[BASE\]** Every attempt, including failures and repair calls, is archived with code, diagnostics, prompts, responses, costs and the decision.

### 7.4 Knowledge source and leakage

- **FR-LLM-11 (M) \[NEW C9\]** Each prompt section shall carry a source tag: `public_docs`, `expert_statement`, `paper_description` or `simulator_source`. The default configuration forbids `simulator_source`.
- **FR-LLM-12 (M) \[NEW C9\]** A leakage audit shall compare prompt text against the environment source tree (token n-gram overlap, n = 8) and report any match. The audit result is printed in the results appendix.
- **FR-LLM-13 (S) \[NEW C9\]** For Gridcraft, an additional run shall use a prompt written by a person who did not write the environment, from public documentation only.
- **FR-LLM-14 (M) \[BASE\]** Human edits to LLM rules shall be saved as new versions with `source = "human_edited"` and counted in the results.

### 7.5 Cost and variance

- **FR-LLM-15 (M) \[NEW C8\]** A cost ledger shall record per call: model id and version, serving format, decoding settings, date, prompt and completion tokens, latency and GPU-seconds. Per-library and per-environment totals are reported alongside final-model training cost.
- **FR-LLM-16 (S) \[NEW C5\]** The full refinement loop shall be repeated with at least 3 LLM sampling seeds per environment, so the variance of the generated library itself is measured, not only the variance of neural training.

## 8. Environment adapters and rule libraries

All four benchmarks shall be documented to the same level as Gridcraft: a schema, a pinned version, a handcrafted library with named individual and joint rules, and an automatically generated rule-library card.

### 8.1 Adapter requirements

- **FR-ENV-1 (M) \[BASE\]** Each adapter maps native observations and actions to the schema, records episodes (joint observations, joint actions, rewards, termination flags, seeds) and reproduces an episode from its seed.
- **FR-ENV-2 (M) \[NEW C3\]** Each adapter shall pin the benchmark repository commit, scenario or layout names, and any wrapper version. These appear in the run manifest.
- **FR-ENV-3 (M) \[NEW C3\]** Each adapter shall document action-resolution semantics used by rules (collision handling, simultaneity, order of effects) and cite where each was taken from (FR-LLM-11 tags apply).
- **FR-ENV-4 (M) \[NEW C3\]** The system shall generate a rule-library card per environment and rule source: number of PSTRs by scope, blocks written, proposed and accepted coverage, conflict rate, support and RDD per rule.
- **FR-ENV-5 (M) \[BASE\]** Predator–Prey uses a cooperative predator team against a fixed, declared prey controller, so the setting remains a Dec-POMDP.

### 8.2 Minimum handcrafted libraries

The table lists the minimum handcrafted rules. Gridcraft rules exist; the others are the proposed minimum to be implemented and verified against the pinned simulator before use.

| Env | Rule id | Scope | Assigns | Key guard / abstention | Origin |
| --- | --- | --- | --- | --- | --- |
| Gridcraft | water | individual | Unchanged known static terrain | Move into observed water; abstain if terrain may change | \[BASE\] |
| Gridcraft | terrain.shift | individual | Terrain translated by established displacement | Known displacement and stable known source; new cells uncovered | \[BASE\] |
| Gridcraft | plank | individual | Wood −1, planks +2 | Craft action, resources, capacity, station | \[BASE\] |
| Gridcraft | collision | joint | Positions of two blocked contenders | Known positions, two contenders, declared resolution rule | \[BASE\] |
| Gridcraft | map | joint | Terrain from shared map | Aligned frames, fresh cached facts, target in view | \[BASE\] |
| Overcooked | pickup | individual | Held object after interacting with a dispenser | Facing dispenser, empty hands | \[NEW C3\] |
| Overcooked | pot.add | individual | Pot ingredient count +1, held object cleared | Facing non-full pot, holding ingredient | \[NEW C3\] |
| Overcooked | pot.timer | individual | Cooking timer advance | Pot cooking, timer below completion | \[NEW C3\] |
| Overcooked | deliver | individual | Held object cleared | Holding soup, facing serving location | \[NEW C3\] |
| Overcooked | blocked.move | individual | Position unchanged, orientation updated | Destination is a counter or wall | \[NEW C3\] |
| Overcooked | player.collision | joint | Both positions unchanged | Same target cell or swap, per declared resolution | \[NEW C3\] |
| Predator–Prey | landmark.rel | individual | Relative landmark positions after own displacement | Own velocity update exact; no contact forces | \[NEW C3\] |
| Predator–Prey | self.vel | individual | Own velocity from damping and action force | No contact within radius margin | \[NEW C3\] |
| Predator–Prey | ally.rel | joint | Relative position of another predator | Both actions known; neither in contact | \[NEW C3\] |
| SMACv2 | unit.type | individual | Own and visible units' type unchanged | Unit visible at t and t+1 | \[NEW C3\] |
| SMACv2 | dead.agent | individual | All-zero observation for a dead agent | Agent health 0 at t | \[NEW C3\] |
| SMACv2 | health.static | joint | Enemy health unchanged | No ally attacks it, no regenerating race, no other damage source visible | \[NEW C3\] |
| SMACv2 | ally.visibility | joint | Ally visibility flag | Both allies' positions and sight range known | \[NEW C3\] |

- **FR-ENV-6 (M) \[NEW C3\]** Each handcrafted rule shall have RDD measured on D\_rev before use. A rule with RDD > τ\_rdd is fixed or removed, and the decision is logged.
- **FR-ENV-7 (M) \[NEW C3, C7\]** Each environment shall have at least one joint rule, so the joint-rule ablation (FR-BASE-6) is defined everywhere.
- **FR-ENV-8 (S) \[NEW C3\]** Low expected coverage (likely in SMACv2) shall be reported, not hidden: the paper table carries coverage next to MSE for every environment.

## 9. Model-based MARL integration

Every world model variant plugs into the same MASAC learner with identical policy objectives and real-interaction budgets, so return differences can be attributed to the world model.

- **FR-MB-1 (M) \[BASE\]** The learner is multi-agent soft actor–critic with centralized recurrent twin critics and local recurrent actors (BenchMARL MASAC/MAMBPO pipeline). A\_MF uses real transitions only; A\_MB adds imagined replay.
- **FR-MB-2 (M) \[BASE\]** Critic target and actor objective:

```latex
y^Q_{i,t} = r_t + \gamma (1 - d_t) \Big[ \min_{k} \bar Q_{i,k}(\eta^j_{t+1}, a'^j) - \alpha_i \log \pi_i(a'_i \mid \eta_{i,t+1}) \Big], \quad J_\pi = \mathbb{E}\big[\alpha_i \log \pi_i(a_i \mid \eta_{i,t}) - \min_k Q_{i,k}(\eta^j_t, (a_i, a_{-i}))\big]
```

- **FR-MB-3 (M) \[BASE\]** Algorithm per iteration: collect real joint transitions; fit observation, reward and termination heads for a fixed number of updates; sample real prefixes; rebuild neural state and symbolic memory; generate imagined branches up to the rollout cap; update policies and critics from a fixed real/imagined replay ratio; evaluate in the real environment.
- **FR-MB-4 (M) \[BASE\]** Entropy coefficient, target averaging rate, burn-in, update counts, rollout cap and replay ratio are identical across variants and stored in the run configuration. Defaults: rollout cap 5, real:imagined ratio 1:1.
- **FR-MB-5 (M) \[BASE\]** Imagined transitions never count toward the real-interaction budget. Evaluation checkpoints every 25,000 real steps up to 10^6.
- **FR-MB-6 (M) \[BASE\]** Reported metrics: return at the final budget, normalized AUC, and interactions to a return threshold reached at three consecutive checkpoints; non-attainment is counted explicitly.
- **FR-MB-7 (S) \[NEW C3\]** The control experiment shall run on Gridcraft and on at least one additional environment (Overcooked recommended, being cheapest), so the downstream claim is not single-environment.
- **FR-MB-8 (S) \[NEW C4\]** The control experiment shall include an external model-based MARL baseline (MAMBA) at the same real-interaction budget, run with its published hyperparameters.
- **FR-MB-9 (M) \[BASE\]** Wall-clock time to each checkpoint shall be logged, so sample efficiency and compute cost can be reported together.

## 10. Baselines, controls and ablations

The experiment grid adds external and symbolic-only baselines, a joint-rule ablation and coverage-matched comparisons, so every headline claim has a control that could have refuted it.

- **FR-BASE-1 (M) \[NEW C4\]** Every baseline shall use the same episode splits, the same evaluation protocol (MSE\_25, covered/uncovered split) and an equal hyperparameter-tuning allowance (default 12 configurations on D\_select).
- **FR-BASE-2 (M) \[NEW C4\]** External baselines shall be wrapped to output the schema's semantic blocks, so the same block error applies. If a baseline predicts in a latent space, its decoder is trained to the schema.

| ID | Condition | Question answered | Concern | Priority | Envs |
| --- | --- | --- | --- | --- | --- |
| B1 | Purely neural JOPM | Reference backbone | — | M | All |
| B2 | MAMBA-style multi-agent world model | Is the JOPM a weak backbone? | C4 | M | All |
| B3 | Centralized Dreamer-style latent model | Same, single-agent family | C4 | S | All |
| B4 | PSWM-only with fallback (FR-PSWM-13) | How much does the neural part add? | C4 | M | All |
| B5 | LLM-written full transition code, no neural part | Is a code world model enough? | C2, C4 | S | Gridcraft, Overcooked |
| B6 | Best strategy on the B2 backbone | Do gains transfer to another backbone? | C1, C4 | S | All |
| A1 | Feature-weighting control | Is Regularization just reweighting? | C1 | M | All |
| A2 | Refinement without RVR (RDD + counterexamples kept) | Does per-rule RVR help? | C1, C5 | M | All |
| A3 | Refinement with RVR only (no RDD) | Which signal matters? | C1 | S | Gridcraft |
| A4 | One-shot initial generation | Value of iteration | — | M | All |
| A5 | Individual-only library (joint rules off) | Value of cross-agent knowledge | C7 | M | All |
| A6 | Joint-only library | Same, reverse | C7 | S | All |
| A7 | λ sweep {0, 0.1, 0.3, 1, 3} | Sensitivity of Regularization | C8 | M | All |
| A8 | Corruption: random values, wrong guards, overclaimed masks, 0–40 % | Robustness to bad rules | — | M | Gridcraft, Overcooked |
| A9 | Coverage-matched evaluation across rule versions | Is the gain from abstention? | C6 | M | All |
| A10 | Number of agents varied (2, 4, 6 where supported) | Does the joint-rule benefit grow with n? | C7 | S | Gridcraft, Predator–Prey |
| A11 | Documentation-only prompt | Knowledge leakage | C9 | S | Gridcraft |

- **FR-BASE-3 (M) \[BASE\]** Iterative refinement conditions share the same candidate budget (4) and the same neural update budget.
- **FR-BASE-4 (M) \[NEW C1\]** The paper's primary claim set shall be tied to A1, A2, A9 and A5 (diagnostics and multi-agent value); strategy comparisons are secondary.
- **FR-BASE-5 (M) \[NEW C6\]** A9 shall be applied to Table-3-style results: each strategy and rule source is also scored on the fixed union block set (FR-DIAG-8).
- **FR-BASE-6 (M) \[NEW C7\]** A5 shall report MSE\_25 overall and on blocks owned by other agents or shared, plus RVR and coverage for the removed joint rules computed on the individual-only model.

## 11. Experiment orchestration, statistics and reporting

Every number in the paper shall be generated from run-level records by a script, with paired statistics and multiple-comparison correction decided before the test set is opened.

### 11.1 Orchestration

- **FR-EXP-1 (M) \[BASE\]** Runs are defined by YAML configuration. A run id is the hash of (configuration, library hash, split hash, seed, code commit).
- **FR-EXP-2 (M) \[BASE\]** Ten seeds per configuration, with separate random streams for initialization, data sampling and environment interaction. Seed i uses the same split and initialization stream across configurations, enabling paired comparisons.
- **FR-EXP-3 (M) \[NEW C8\]** Offline datasets shall be described and reported per environment: collection policies (mixture of random, scripted and partially trained MASAC policies, with proportions), number of episodes and transitions per split, and episode length. Default minimum: 1,000 episodes per environment.
- **FR-EXP-4 (M) \[BASE\]** Prediction uses H = 25 open-loop rollouts from held-out prefixes with replayed joint actions. Per-step error curves and per-run 25-step averages are both stored.
- **FR-EXP-5 (M) \[BASE\]** Training measurements: seconds per update, updates and seconds to a validation target (three consecutive checkpoints), inference latency per joint step. All timings on one declared hardware profile, warm-up excluded.
- **FR-EXP-6 (M) \[NEW C8\]** Total development cost = final-model training + LLM generation and refinement (ledger, FR-LLM-15) + candidate training. It is reported per environment.

### 11.2 Statistics

- **FR-STAT-1 (M) \[NEW C5\]** The independent unit is the seed. Results report mean, SD and a 95 % bootstrap confidence interval (10,000 resamples, percentile) of the mean.
- **FR-STAT-2 (M) \[NEW C5\]** Pairwise comparisons use the paired difference across seeds: Wilcoxon signed-rank test (primary) and paired t-test (secondary), plus Hedges' g and a bootstrap CI of the mean difference. Unpaired comparisons use Welch's t-test.
- **FR-STAT-3 (M) \[NEW C5\]** Comparisons are declared in advance per research question (RQ1 prediction, RQ2 refinement, RQ3 robustness/cost, RQ4 control). Holm–Bonferroni correction is applied within each RQ family.
- **FR-STAT-4 (M) \[NEW C5\]** The report generator shall enforce wording: "outperforms" or "reduces" only when the Holm-adjusted p < 0.05; otherwise "lower mean, not significant". "Confirms" is never generated.
- **FR-STAT-5 (S) \[NEW C5\]** If a pre-declared primary comparison has a CI width above a declared tolerance after 10 seeds, the protocol allows extension to 20 seeds for all arms of that comparison; the extension rule is fixed before test access.
- **FR-STAT-6 (M) \[BASE\]** Threshold-crossing statistics use attained runs only and report attainment counts; no extrapolation beyond the budget.

### 11.3 Reporting

- **FR-REP-1 (M) \[BASE\]** `raw_runs.csv` (one row per experiment, configuration, run, checkpoint, metric) is the single source; all tables and figures are rebuilt from it by one command.
- **FR-REP-2 (M) \[NEW C10\]** Table notes shall be generated from metadata: when a Projection-pre row equals the purely neural row, the note states that Projection trains the same network on the same data and seeds; structural zero RVR rows carry the "structural" label.
- **FR-REP-3 (M) \[NEW C10\]** For version traces, the table states which library was used for MSE: the full final library with only the traced rule swapped to the listed version. The harness shall build those swapped libraries automatically.
- **FR-REP-4 (M) \[NEW C3, C6\]** The main prediction table shall carry, per environment, coverage and covered/uncovered error next to MSE\_25, plus the significance marks from FR-STAT-3.
- **FR-REP-5 (S) \[NEW C1\]** A rule-level figure (from FR-DIAG-14) shall be producible for the paper: RVR per model vs RDD per rule, sized by support.

## 12. Reproducibility, provenance and anonymization

The artifact package shall let an auditor go from any number in the paper to the exact configuration, data split, rule versions, prompts and seed that produced it; this closes the provenance gap stated in the current draft.

### 12.1 Run manifest

- **FR-REPRO-1 (M) \[NEW C8\]** Each run writes a manifest: run id, code commit, configuration, library hash, split hash, seeds, benchmark commits, hardware, start/end times, and output file hashes.
- **FR-REPRO-2 (M) \[NEW C8\]** Each row of `raw_runs.csv` carries its run id, so measurements map to configurations without manual bookkeeping.

### 12.2 Rule and LLM provenance

- **FR-REPRO-3 (M) \[BASE\]** All rule versions, including rejected and failed candidates and human edits, are archived with code hash, parent version and evidence ids.
- **FR-REPRO-4 (M) \[NEW C8, C9\]** Full prompts and responses, decoding settings, dates, token counts and the knowledge-source tags are archived per call.

### 12.3 Determinism

- **FR-REPRO-5 (S) \[BASE\]** PyTorch deterministic algorithms shall be enabled where available. Rerunning a run on the same hardware shall reproduce MSE\_25 within 1 % relative; differences beyond that are logged as a reproducibility defect.
- **FR-REPRO-6 (M) \[BASE\]** The PSWM engine is deterministic: the same context and library always produce the same output.

### 12.4 Anonymized export

- **FR-REPRO-7 (M) \[NEW\]** An export command shall build a double-blind package: it strips user names, home paths, git remotes and author metadata from code, configs, CSVs, notebooks and PDFs, and replaces own-repository URLs (e.g. the Gridcraft repository) with an anonymous mirror URL.
- **FR-REPRO-8 (M) \[NEW\]** The export shall fail if a deny-list of strings (author names, usernames, institution names, own GitHub handle) appears anywhere in the package.
- **FR-REPRO-9 (M) \[NEW C3\]** References to external benchmarks shall carry pinned commits; placeholders such as "commit remains to be pinned" shall make the export fail.

## 13. Non-functional requirements

The system shall be safe to run LLM-written code, fast enough for the full experiment grid on a small GPU cluster, and extensible to new environments without touching the core.

| ID | Category | Requirement | Priority |
| --- | --- | --- | --- |
| NFR-SEC-1 | Security | LLM rule code runs in a restricted namespace: no imports beyond an allow-list (math, numpy), no I/O, no network, CPU and time limit per call (default 50 ms per context) | M |
| NFR-SEC-2 | Security | Simulator objects are never passed to rule code; only frozen schema-level views | M |
| NFR-PERF-1 | Performance | PSWM execution adds ≤ 1 ms per joint step at batch 1 on Gridcraft (current: +0.35 ms for Projection) | S |
| NFR-PERF-2 | Performance | Batched diagnostic evaluation of 10^5 contexts per rule library completes in ≤ 10 min on one CPU node | S |
| NFR-PERF-3 | Performance | The full prediction grid (4 envs × all conditions × 10 seeds) fits in ≤ 2,000 GPU-hours; the planner reports the estimate before launch | S |
| NFR-EXT-1 | Extensibility | A new environment needs only a schema, an adapter and a rule library; no core code changes | M |
| NFR-EXT-2 | Extensibility | A new backbone needs only the FR-NN-1 interface | M |
| NFR-EXT-3 | Extensibility | A new integration strategy registers through a plugin entry point | S |
| NFR-QUAL-1 | Quality | ≥ 90 % line coverage on the rule engine and diagnostics modules | M |
| NFR-QUAL-2 | Quality | Type-checked (mypy strict) public interfaces | S |
| NFR-USE-1 | Usability | One command per stage: `collect`, `generate-rules`, `refine`, `train`, `evaluate`, `control`, `report`, `export-anon` | M |
| NFR-DOC-1 | Documentation | Each environment ships a rule-library card and a semantics note (FR-ENV-3, FR-ENV-4) | M |

### 13.1 Future extensions (outside this release)

Stochastic symbolic effects with distributional targets; inequality constraints through a separate constraint mechanism; decentralized symbolic prediction with communication; a user study of rule-level interpretability. Interfaces should not preclude them: `Assignment.value` may later become a distribution and `RuleSpec` may later gain a `kind = "constraint"`.

## 14. Verification and acceptance tests

The re-implementation is accepted when all tests below pass and the Gridcraft results of the current draft are reproduced within their confidence intervals.

| Test | Verifies | Method | Pass criterion |
| --- | --- | --- | --- |
| T-01 | FR-IF-1 | Guard returns UNKNOWN | Empty mask, no proposal |
| T-02 | FR-PSWM-6 | Two rules assign different categories to one block | Block uncovered; both proposals in conflict log; counters updated |
| T-03 | FR-PSWM-11 | Shuffle rule order within stages, 3 orders | Identical (Y, M) |
| T-04 | FR-SCH-2 | One-hot block with 1 wrong entry vs 3 wrong entries | Each counts as one violation |
| T-05 | FR-IF-2, NFR-SEC-1 | Rule code tries to import os, write a file, mutate context | Load rejected or call aborted; context unchanged |
| T-06 | FR-DIAG-4 | Projection and Residual with R\_eval = enforced library | RVR\_out = 0 and labelled structural |
| T-07 | FR-DIAG-1 | Rule with zero support | Reported N/A |
| T-08 | FR-INT-4 | Residual input with assigned 0 vs abstention | Different network inputs |
| T-09 | FR-INT-7 | λ = 0 | Regularization equals purely neural within 1e−6 on same seed |
| T-10 | FR-INT-10 | Symbolic targets equal to data | Regularization and feature-weighting losses coincide |
| T-11 | FR-PSWM-2 | Fact older than max age | Fact expired; dependent rule abstains |
| T-12 | FR-LLM-2 | Scan all prompts and selection inputs for D\_test episode ids and transitions | Zero matches |
| T-13 | FR-LLM-12 | Insert a simulator-source snippet into a prompt | Leakage audit flags it |
| T-14 | FR-STAT-2/3 | Compare statistics module to SciPy on fixed data | Identical p-values to 1e−9; Holm order correct |
| T-15 | FR-STAT-4 | Non-significant difference in a generated table | Text says "lower mean, not significant" |
| T-16 | FR-REPRO-8 | Plant an author name in a CSV | Export fails with location |
| T-17 | FR-DIAG-7/8 | Synthetic case with known covered and uncovered errors | Decomposition sums to MSE\_H; fixed-set MSE independent of mask |
| T-18 | FR-PSWM-12 | Disable joint rules by scope filter | Only individual proposals; library hash changes |
| T-19 | Regression | Re-run Gridcraft neural, three strategies × refined rules, 10 seeds | Each mean MSE\_25 within the original 95 % CI (e.g. Regularization refined 0.055) |
| T-20 | Regression | Re-run terrain.shift trace v0–v3 | RVR, RDD, coverage within original CIs |

- **FR-VER-1 (M)** Tests T-01 to T-18 run in CI on every commit; T-19 and T-20 run before each paper-results freeze.
- **FR-VER-2 (M)** A release is accepted only if every "M" requirement in Sections 3–12 is traced to at least one passing test or a generated report artifact.

## 15. Traceability: review concerns to requirements

Each of the ten review concerns maps to at least one mandatory requirement and one artifact the paper can show.

| Concern | Requirements | Paper evidence produced |
| --- | --- | --- |
| C1 Novelty of strategies | FR-BASE-4, FR-DIAG-14, FR-REP-5, A1–A3, B6 | Diagnostics framed as primary contribution; rule-level figure; strategies shown on two backbones |
| C2 Related-work gaps | B5 (code world model baseline); references in 1.5 | Comparison with code-only world model; MAMBA, APHYNITY, Dainese et al. cited |
| C3 Black-box benchmarks | FR-ENV-2 to 8, FR-MB-7, FR-REP-4, FR-REPRO-9 | Rule-library card per environment; coverage and RDD in main table; second control environment |
| C4 Weak-backbone doubt | FR-NN-1, FR-PSWM-13, FR-BASE-1/2, B2–B6, FR-MB-8 | External world-model, PSWM-only and code-only baselines |
| C5 Statistics | FR-STAT-1 to 5, FR-LLM-16 | CIs, paired tests, Holm correction, controlled wording, LLM-seed variance |
| C6 Covered/uncovered split | FR-PSWM-9, FR-DIAG-6 to 8, FR-BASE-5, A9 | Error split, conflict rates, coverage-matched comparison |
| C7 Thin multi-agent aspect | FR-SCH-4, FR-PSWM-12, FR-DIAG-12, FR-ENV-7, FR-BASE-6, A5, A6, A10 | Joint-rule ablation in all environments; errors on other-agent blocks; scaling with n |
| C8 Missing details | FR-NN-3/4, FR-INT-9, FR-LLM-15, FR-EXP-3, FR-EXP-6, FR-REPRO-1/2/4 | Architecture, update definition, λ sweep, dataset sizes, LLM cost, total development cost |
| C9 Knowledge leakage | FR-LLM-11 to 13, A11 | Source tags, leakage audit, documentation-only prompt run |
| C10 Table clarity | FR-REP-2, FR-REP-3 | Generated table notes; automatic swapped-library construction for version traces |
| Anonymity (Gridcraft URL) | FR-REPRO-7, FR-REPRO-8 | Anonymized export with deny-list check |

## Appendix A. Configuration and data schemas

### A.1 Run configuration (example)

```yaml
run:
  experiment: prediction          # prediction | refinement | robustness | cost | control
  env: gridcraft
  env_commit: <pinned>
  seed: 3
  split_manifest: splits/gridcraft_v1.json
world_model:
  backbone: jopm_lstm             # jopm_lstm | mamba_wm | dreamer_central | pswm_only | llm_code
  strategy: regularization        # none | projection | residual | regularization | feature_weighting
  lambda: 1.0
  sizes: {enc: [256, 256], lstm: 256, g: [256, 256], dec: [256, 256]}
  optim: {name: adam, lr: 3.0e-4, batch_seqs: 32, seq_len: 50, burn_in: 10, updates: 20000}
rules:
  library: libraries/gridcraft/refined_v3.json
  filter: {scope: [individual, joint], disable_ids: []}
  conflict_policy: block_rejection
  diagnostic_library: libraries/gridcraft/refined_v3.json
evaluation:
  horizon: 25
  error_split: true
  fixed_block_set: union_of_versions
llm:
  model: qwen3.6-35b-a3b-nvfp4
  temperature: 0.2
  max_candidates: 4
  token_budget: 400000
  allowed_sources: [public_docs, expert_statement, paper_description]
stats:
  ci: bootstrap_percentile
  resamples: 10000
  test: wilcoxon
  correction: holm
```

### A.2 raw\_runs.csv

```csv
run_id,experiment,env,config_hash,library_hash,split_hash,seed,checkpoint,metric,value,support,unit
7f3a…,prediction,gridcraft,c19e…,a02b…,5d1c…,3,final,mse25,0.0551,,normalized
7f3a…,prediction,gridcraft,c19e…,a02b…,5d1c…,3,final,mse25_covered,0.0213,,normalized
7f3a…,prediction,gridcraft,c19e…,a02b…,5d1c…,3,final,rvr.terrain.shift.v3,0.040,11471,fraction
```

### A.3 Feedback record sent to P3

```json
{
  "evidence_id": "rev-000412",
  "rule_id": "terrain.shift", "rule_version": 0,
  "context_origin": "real",
  "action": {"agent_0": "move_east"},
  "guard": "TRUE",
  "assigned_blocks": ["agent0.terrain[*]"],
  "symbolic": "shifted terrain",
  "neural": "unchanged terrain",
  "observed": "unchanged terrain",
  "rule_stats": {"support": 17202, "rvr_neural": 0.31, "rvr_reg": 0.32, "rdd": 0.248, "conflicts": 0}
}
```

### A.4 Other files

| File | Content |
| --- | --- |
| `splits/<env>_<v>.json` | Episode ids per split, seeds, hash |
| `libraries/<env>/<name>.json` | Library manifest with rule versions and code hashes |
| `rules/<env>/<rule_id>/v<k>.py` | Rule source per version |
| `llm/calls.jsonl` | One line per LLM call: prompt, response, tokens, latency, source tags |
| `llm/candidates.jsonl` | Candidate archive: validation, diagnostics, selection decision |
| `diagnostics/<run_id>.json` | Per-rule report with counterexamples |
| `manifests/<run_id>.json` | Run manifest (FR-REPRO-1) |