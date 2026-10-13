"""Locked synthetic statistical rules using the existing CPU PyTorch dependency."""
from dataclasses import dataclass
import ctypes
import hashlib
import torch

REPLICATES, SEED = 20000, 6010


@dataclass(frozen=True)
class ResamplingBank:
    groups: tuple
    indices: tuple
    replicates: int
    seed: int
    identity: str


def make_bank(strata, replicates=REPLICATES, seed=SEED):
    strata = torch.as_tensor(strata, device="cpu")
    if strata.ndim != 1 or not len(strata):
        raise ValueError("Nonempty case strata required")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    groups, indices = [], []
    digest = hashlib.sha256()
    digest.update(str((seed, replicates, strata.tolist(), torch.__version__)).encode())
    for value in sorted(set(strata.tolist())):
        members = torch.nonzero(strata == value).flatten()
        draws = members[torch.randint(len(members), (replicates, len(members)), generator=rng)]
        groups.append(members)
        indices.append(draws)
        digest.update(ctypes.string_at(draws.data_ptr(), draws.numel()*draws.element_size()))
    return ResamplingBank(tuple(groups), tuple(indices), replicates, seed, digest.hexdigest())


def stratified_means(values, bank):
    values = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite synthetic observation")
    if values.shape[0] != sum(len(g) for g in bank.groups):
        raise ValueError("Case schema differs from sealed resampling bank")
    point = torch.stack([values[g].mean(dim=0) for g in bank.groups]).mean(dim=0)
    draws = torch.stack([values[g].mean(dim=1) for g in bank.indices]).mean(dim=0)
    return point, draws


def quantile(values, probability):
    return torch.quantile(torch.as_tensor(values, dtype=torch.float64), probability,
                          dim=0, interpolation="linear")


def ablation_rule(unablated, ablated, bank):
    if len(bank.groups) != 4:
        raise ValueError("Exactly four length/parity strata required")
    u, a = torch.as_tensor(unablated), torch.as_tensor(ablated)
    if u.shape != a.shape or u.ndim != 2 or u.shape[1] != 4:
        raise ValueError("Exactly four paired arm/program contrasts required")
    if not ((u == 0) | (u == 1)).all() or not ((a == 0) | (a == 1)).all():
        raise ValueError("Robust-case indicators must be binary")
    point, draws = stratified_means(u.to(torch.float64) - a, bank)
    lower = quantile(draws, .0125)
    return dict(point=point, lower=lower, passes=(point >= .05) & (lower > 0),
                family_size=4, quantile=.0125, bank=bank.identity)


def latency_rule(timed, lengths, parities, bank):
    """timed[case, arm(T,C), program(0,1), repetition(5)]. Warm-up absent."""
    t = torch.as_tensor(timed, dtype=torch.float64, device="cpu")
    if t.ndim != 4 or t.shape[1:] != (2, 2, 5):
        raise ValueError("Missing/extra prescribed timed measurements; exactly five required")
    if not torch.isfinite(t).all():
        raise ValueError("Unassessable nonfinite latency; no measurement discarded")
    lengths, parities = torch.as_tensor(lengths), torch.as_tensor(parities)
    if lengths.shape != (len(t),) or parities.shape != (len(t),):
        raise ValueError("Profiling case metadata mismatch")
    if len(bank.groups) != 4:
        raise ValueError("Exactly four profiling length/parity strata required")
    means = t.mean(dim=-1)
    points, uppers = [], []
    for program in (0, 1):
        for length in (16, 24):
            selections = [torch.nonzero((lengths == length) & (parities == p)).flatten() for p in (0, 1)]
            if any(not len(g) for g in selections):
                raise ValueError("Missing profiling stratum")
            selected_bank_indices = []
            for group in selections:
                match = [d for g, d in zip(bank.groups, bank.indices) if torch.equal(g, group)]
                if len(match) != 1:
                    raise ValueError("Profiling bank must preserve length/parity case identity")
                selected_bank_indices.append(match[0])
            point = torch.stack([means[g, :, program].mean(dim=0) for g in selections]).mean(dim=0)
            draws = torch.stack([means[d, :, program].mean(dim=1) for d in selected_bank_indices]).mean(dim=0)
            if not torch.isfinite(point).all() or not torch.isfinite(draws).all():
                raise ValueError("Unassessable nonfinite arithmetic means")
            if point[1] <= 0 or (draws[:, 1] <= 0).any():
                raise ValueError("Unassessable nonpositive denominator")
            ratios = draws[:, 0] / draws[:, 1]
            if not torch.isfinite(ratios).all():
                raise ValueError("Unassessable nonfinite latency ratio")
            points.append(point[0] / point[1])
            uppers.append(quantile(ratios, .9875))
    upper = torch.stack(uppers)
    return dict(order=["P0/L16", "P0/L24", "P1/L16", "P1/L24"],
                point=torch.stack(points), upper=upper, passes=upper <= 1.10,
                family_size=4, quantile=.9875, bank=bank.identity)
