from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.integrate import trapezoid


def bootstrap(values, resamples=10000, seed=0):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Statistics require finite seed-level values")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start)
        means[start:start + count] = rng.choice(values, (count, len(values)), replace=True).mean(1)
    return np.quantile(means, [.025, .975]).tolist()


def summarize(values, **kwargs):
    values = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "ci95": bootstrap(values, **kwargs)}


def compare(left, right, *, paired=True, **kwargs):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if min(len(a), len(b)) < 2:
        raise ValueError("Comparisons need at least two independent seeds")
    if paired:
        if a.shape != b.shape:
            raise ValueError("Paired comparisons require matched seeds")
        delta = a - b
        all_zero = bool(np.all(delta == 0))
        primary = 1. if all_zero else float(stats.wilcoxon(a, b).pvalue)
        secondary = 1. if all_zero else float(stats.ttest_rel(a, b).pvalue)
        ci = bootstrap(delta, **kwargs)
        # Paired Hedges g_av uses the average marginal variance.
        sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
        df = 2 * len(a) - 2
    else:
        primary = secondary = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        rng = np.random.default_rng(kwargs.get("seed", 0))
        diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(kwargs.get("resamples", 10000))]
        ci = np.quantile(diffs, [.025, .975]).tolist()
        df = len(a) + len(b) - 2
        sd = float(np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / df))
    difference = float(a.mean() - b.mean())
    return {"difference": difference, "ci95": ci, "p_primary": primary, "p_secondary": secondary,
            "hedges_g_av" if paired else "hedges_g": (1 - 3 / (4 * df - 1)) * difference / sd if sd else None}


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-values")
    order = np.argsort(p, kind="stable")
    corrected = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    out = np.empty_like(p)
    out[order] = corrected
    return out.tolist()


def wording(difference, adjusted_p, *, lower_is_better=True):
    better = difference < 0 if lower_is_better else difference > 0
    if adjusted_p < .05 and better:
        return "reduces" if lower_is_better else "outperforms"
    return "lower mean, not significant" if difference < 0 else "higher mean, not significant"


def threshold_crossing(checkpoints, values, threshold, *, lower_is_better=True, consecutive=3):
    streak = 0
    for point, value in zip(checkpoints, values):
        hit = value <= threshold if lower_is_better else value >= threshold
        streak = streak + 1 if hit else 0
        if streak >= consecutive:
            return point
    return None


def control_metrics(steps, returns, threshold):
    if len(steps) != len(returns) or len(steps) < 2:
        raise ValueError("At least two control checkpoints are required")
    return {"final_return": float(returns[-1]),
            "normalized_auc": float(trapezoid(returns, steps) / (steps[-1] - steps[0])),
            "interactions_to_threshold": threshold_crossing(steps, returns, threshold, lower_is_better=False)}


def extension_decision(arms, protocol):
    """No optional stopping: all arms extend together under a frozen rule."""
    if not protocol.get('enabled',False): return {'extend':False,'reason':'disabled'}
    tolerance=protocol.get('ci_width_tolerance',0)
    if tolerance<=0: raise ValueError('A positive CI-width tolerance must be declared')
    initial=protocol.get('initial_seeds',10); maximum=protocol.get('extended_seeds',20)
    if initial!=10 or maximum!=20: raise ValueError('The SRS extension is 10 to 20 seeds')
    if len(arms)<2 or any(len(v)!=initial for v in arms.values()):
        return {'extend':False,'reason':'requires ten seeds in every arm'}
    if len({tuple(sorted(v)) for v in arms.values()})!=1: raise ValueError('Extension arms must have paired seeds')
    names=sorted(arms); reference=arms[names[0]]; widths={}
    for name in names[1:]:
        ci=bootstrap([arms[name][s]-reference[s] for s in sorted(reference)])
        widths[name]=ci[1]-ci[0]
    extend=any(width>tolerance for width in widths.values())
    return {'extend':extend,'ci_widths':widths,'arms':names,'additional_seeds':list(range(10,20)) if extend else []}


def reproducibility(reference, repeated, tolerance=.01):
    error=abs(float(repeated)-float(reference))
    relative=error/max(abs(float(reference)),1e-12)
    return {'reference':reference,'repeated':repeated,'relative_error':relative,'tolerance':tolerance,
            'passed':relative<=tolerance,'status':'passed' if relative<=tolerance else 'reproducibility_defect'}
