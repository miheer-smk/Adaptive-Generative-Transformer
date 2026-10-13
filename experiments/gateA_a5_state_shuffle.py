from __future__ import annotations

import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.agt import AGTv3
from data.task_a import TaskAParity


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIM = 64
CKPT = "experiments/runs/stage1_parity_seed3305_best.pt"

SEEDS = {
    8: 5808,
    128: 5128,
}


def load_model():
    model = AGTv3(
        dim=DIM,
        num_memory_slots=16,
        vocab_size=2,
    ).to(DEVICE)

    checkpoint = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def make_bits(length: int):
    if length == 8:
        return torch.tensor(
            list(itertools.product([0, 1], repeat=8)),
            dtype=torch.long,
            device=DEVICE,
        )

    g = torch.Generator()
    g.manual_seed(SEEDS[length])

    return torch.randint(
        0,
        2,
        (512, length),
        generator=g,
        dtype=torch.long,
        device="cpu",
    ).to(DEVICE)


def make_derangement(batch_size: int, seed: int):
    if batch_size < 2:
        raise RuntimeError("Need batch size >= 2 for state shuffle.")

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    base = torch.arange(batch_size)

    # Search deterministically for a non-identity permutation.
    for _ in range(100):
        perm = torch.randperm(batch_size, generator=g)

        if not torch.equal(perm, base) and torch.all(perm != base):
            return perm.to(DEVICE)

    # Deterministic fallback.
    return torch.roll(base, shifts=1).to(DEVICE)


def shuffle_one_state(state, gdn_index: int, perm: torch.Tensor):
    """
    Shuffle only the persistent recurrent state S across examples.

    Nothing else is touched:
      - target GDN convolution buffer is preserved
      - non-target GDN state is preserved
      - non-target GDN convolution buffer is preserved
    """
    target = state.ctx.gdn[gdn_index]
    other = state.ctx.gdn[5 if gdn_index == 4 else 4]

    before_target = target.committed_state.detach().clone()
    before_target_conv = (
        None
        if target.committed_conv_buffer is None
        else target.committed_conv_buffer.detach().clone()
    )

    before_other = (
        None
        if other.committed_state is None
        else other.committed_state.detach().clone()
    )
    before_other_conv = (
        None
        if other.committed_conv_buffer is None
        else other.committed_conv_buffer.detach().clone()
    )

    target.committed_state.copy_(
        target.committed_state.index_select(0, perm)
    )

    if torch.equal(before_target, target.committed_state):
        raise AssertionError("State shuffle produced no change.")

    if before_target_conv is not None and not torch.equal(
        before_target_conv,
        target.committed_conv_buffer,
    ):
        raise AssertionError(
            "Target GDN convolution buffer changed during state shuffle."
        )

    if before_other is not None and not torch.equal(
        before_other,
        other.committed_state,
    ):
        raise AssertionError(
            "Non-target GDN state changed during state shuffle."
        )

    if before_other_conv is not None and not torch.equal(
        before_other_conv,
        other.committed_conv_buffer,
    ):
        raise AssertionError(
            "Non-target GDN convolution buffer changed during state shuffle."
        )

    return {
        "target_shape": list(before_target.shape),
        "target_numel_per_example": int(before_target[0].numel()),
        "permutation_changed_all_examples": bool(
            torch.all(perm != torch.arange(perm.numel(), device=perm.device))
        ),
        "target_conv_unchanged": True,
        "other_state_unchanged": True,
        "other_conv_unchanged": True,
    }


@torch.no_grad()
def run_condition(bits, gdn_index=None, perm=None):
    model = load_model()

    x = task.token_features[bits]
    batch_size = bits.shape[0]

    state = model.initial_persistent_state(
        batch_size=batch_size,
        device=DEVICE,
        dtype=x.dtype,
    )

    # Validated Stage-1 API: C_prev remains an explicit compatibility input
    # and is not updated from the returned persistent state.
    C_prev = torch.zeros(
        batch_size,
        model.dim,
        device=DEVICE,
        dtype=x.dtype,
    )

    records = []

    for t in range(bits.shape[1]):
        out = model.step(
            x[:, t, :],
            state,
            C_prev,
        )

        state = out["state"]

        # S_t affects token t+1, so intervene only at inter-token boundaries.
        if (
            gdn_index is not None
            and perm is not None
            and t < bits.shape[1] - 1
        ):
            rec = shuffle_one_state(
                state,
                gdn_index=gdn_index,
                perm=perm,
            )
            records.append(rec)

    logits = out["logits"]
    target = bits.sum(dim=1) % 2

    ce = F.cross_entropy(logits, target).item()
    acc = (logits.argmax(dim=-1) == target).float().mean().item()

    return {
        "loss": ce,
        "accuracy": acc,
        "intervention_count": len(records),
        "records": records,
        "final_logit_shape": list(logits.shape),
    }


def summarize(records):
    if not records:
        return {}

    return {
        "num_interventions": len(records),
        "target_shape": records[0]["target_shape"],
        "target_numel_per_example": records[0]["target_numel_per_example"],
        "permutation_changed_all_examples": all(
            r["permutation_changed_all_examples"] for r in records
        ),
        "target_conv_unchanged": all(
            r["target_conv_unchanged"] for r in records
        ),
        "other_state_unchanged": all(
            r["other_state_unchanged"] for r in records
        ),
        "other_conv_unchanged": all(
            r["other_conv_unchanged"] for r in records
        ),
    }


task = TaskAParity(dim=DIM, device=DEVICE)

print("=" * 96)
print("GATE A5 — PERSISTENT GDN STATE SHUFFLE")
print("=" * 96)
print(f"device={DEVICE}")
print(f"dim={DIM}")
print(f"checkpoint={CKPT}")
print("target = state.ctx.gdn[i].committed_state")
print("shuffle = fixed batch permutation at every inter-token boundary")
print("permutation = non-identity derangement")
print("preserved = target convolution buffer + non-target GDN state/buffer")
print("final-token shuffle = omitted")
print("lengths = L8, L128")
print()

results = {}
permutation_audit = {}

for length in (8, 128):
    bits = make_bits(length)
    B = bits.shape[0]

    print("-" * 96)
    print(f"LENGTH {length}  examples={B}")
    print("-" * 96)

    clean = run_condition(bits)
    results[f"L{length}_CLEAN"] = {
        "loss": clean["loss"],
        "accuracy": clean["accuracy"],
        "intervention_summary": {},
    }

    print(
        f"CLEAN          CE={clean['loss']:.6f} "
        f"ACC={clean['accuracy']:.4f}"
    )

    for gdn_index, gdn_name, seed in (
        (4, "GDN1", SEEDS[length] + 400),
        (5, "GDN2", SEEDS[length] + 500),
    ):
        perm = make_derangement(B, seed)

        permutation_audit[f"L{length}_{gdn_name}"] = {
            "seed": seed,
            "batch_size": B,
            "permutation": perm.cpu().tolist(),
        }

        result = run_condition(
            bits,
            gdn_index=gdn_index,
            perm=perm,
        )

        summary = summarize(result["records"])
        key = f"L{length}_{gdn_name}_SHUFFLE"

        results[key] = {
            "loss": result["loss"],
            "accuracy": result["accuracy"],
            "intervention_summary": summary,
        }

        print(
            f"{gdn_name + '_SHUFFLE':<14} "
            f"CE={result['loss']:.6f} "
            f"ACC={result['accuracy']:.4f} "
            f"ΔACC={result['accuracy'] - clean['accuracy']:+.4f} "
            f"interventions={summary['num_interventions']} "
            f"shape={summary['target_shape']} "
            f"all_examples_moved={summary['permutation_changed_all_examples']} "
            f"target_conv_ok={summary['target_conv_unchanged']} "
            f"other_state_ok={summary['other_state_unchanged']}"
        )

        assert summary["target_conv_unchanged"]
        assert summary["other_state_unchanged"]
        assert summary["other_conv_unchanged"]
        assert summary["permutation_changed_all_examples"]


report = {
    "gate": "A5",
    "checkpoint": CKPT,
    "device": str(DEVICE),
    "dim": DIM,
    "target": "persistent committed recurrent state S",
    "target_paths": {
        "GDN1": "state.ctx.gdn[4].committed_state",
        "GDN2": "state.ctx.gdn[5].committed_state",
    },
    "shuffle_definition": (
        "fixed batch derangement applied to the selected GDN committed state "
        "after each token except the final token"
    ),
    "preserved": [
        "selected GDN convolution buffer",
        "non-target GDN committed state",
        "non-target GDN convolution buffer",
        "C_prev compatibility input",
    ],
    "results": results,
    "permutation_audit": permutation_audit,
}

Path("experiments/gateA_a5_state_shuffle.json").write_text(
    json.dumps(report, indent=2)
)

print("\n" + "=" * 96)
print("=== GATE A5 COMPLETE ===")
print("WROTE experiments/gateA_a5_state_shuffle.json")
print("=" * 96)
