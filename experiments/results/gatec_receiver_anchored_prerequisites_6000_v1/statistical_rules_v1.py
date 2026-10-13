"""Locked acceptance arithmetic. Prerequisites call this on SYNTHETIC arrays only."""
from dataclasses import dataclass
import hashlib
import numpy as np

REPLICATES, SEED = 20000, 6010


@dataclass(frozen=True)
class ResamplingBank:
    groups: tuple
    indices: tuple
    replicates: int
    seed: int
    identity: str


def make_bank(strata, replicates=REPLICATES, seed=SEED):
    strata = np.asarray(strata)
    if strata.ndim != 1 or not len(strata):
        raise ValueError("Nonempty case strata required")
    rng = np.random.default_rng(seed)
    groups, indices = [], []
    digest = hashlib.sha256()
    digest.update(str((seed, replicates, strata.tolist())).encode())
    for value in sorted(set(strata.tolist())):
        members = np.flatnonzero(strata == value)
        draws = members[rng.integers(0, len(members), size=(replicates, len(members)))]
        groups.append(members)
        indices.append(draws)
        digest.update(draws.tobytes())
    return ResamplingBank(tuple(groups), tuple(indices), replicates, seed, digest.hexdigest())


def stratified_means(values, bank):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite synthetic observation")
    if values.shape[0] != sum(len(g) for g in bank.groups):
        raise ValueError("Case schema differs from sealed resampling bank")
    point = np.stack([values[g].mean(axis=0) for g in bank.groups]).mean(axis=0)
    draws = np.stack([values[g].mean(axis=1) for g in bank.indices]).mean(axis=0)
    return point, draws


def quantile(values, probability):
    return np.quantile(values, probability, axis=0, method="linear")


def ablation_rule(unablated, ablated, bank):
    u, a = np.asarray(unablated), np.asarray(ablated)
    if u.shape != a.shape or u.ndim != 2 or u.shape[1] != 4:
        raise ValueError("Exactly four paired arm/program contrasts required")
    if not np.isin(u, [0, 1]).all() or not np.isin(a, [0, 1]).all():
        raise ValueError("Robust-case indicators must be binary")
    point, draws = stratified_means(u.astype(np.float64) - a, bank)
    lower = quantile(draws, .0125)
    return dict(point=point, lower=lower, passes=(point >= .05) & (lower > 0),
                family_size=4, quantile=.0125, bank=bank.identity)


def latency_rule(timed, lengths, parities, bank):
    """timed[case, arm(T,C), program(0,1), repetition(5)]. Warm-up absent."""
    t = np.asarray(timed, dtype=np.float64)
    if t.ndim != 4 or t.shape[1:] != (2, 2, 5):
        raise ValueError("Missing/extra prescribed timed measurements; exactly five required")
    if not np.isfinite(t).all():
        raise ValueError("Unassessable nonfinite latency; no measurement discarded")
    lengths, parities = np.asarray(lengths), np.asarray(parities)
    if lengths.shape != (len(t),) or parities.shape != (len(t),):
        raise ValueError("Profiling case metadata mismatch")
    means = t.mean(axis=-1)
    points, uppers = [], []
    for program in (0, 1):
        for length in (16, 24):
            selections = [np.flatnonzero((lengths == length) & (parities == p)) for p in (0, 1)]
            if any(not len(g) for g in selections):
                raise ValueError("Missing profiling stratum")
            selected_bank_indices = []
            for group in selections:
                match = [d for g, d in zip(bank.groups, bank.indices) if np.array_equal(g, group)]
                if len(match) != 1:
                    raise ValueError("Profiling bank must preserve length/parity case identity")
                selected_bank_indices.append(match[0])
            point = np.stack([means[g, :, program].mean(axis=0) for g in selections]).mean(axis=0)
            draws = np.stack([means[d, :, program].mean(axis=1) for d in selected_bank_indices]).mean(axis=0)
            if point[1] <= 0 or np.any(draws[:, 1] <= 0):
                raise ValueError("Unassessable nonpositive denominator")
            ratios = draws[:, 0] / draws[:, 1]
            points.append(point[0] / point[1])
            uppers.append(float(quantile(ratios, .9875)))
    return dict(order=["P0/L16", "P0/L24", "P1/L16", "P1/L24"],
                point=np.asarray(points), upper=np.asarray(uppers), passes=np.asarray(uppers) <= 1.10,
                family_size=4, quantile=.9875, bank=bank.identity)
