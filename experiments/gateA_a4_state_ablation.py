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
    8: 4808,
    128: 4128,
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


def clone_tensor(x):
    return None if x is None else x.detach().clone()


def intervene_state(state, target_gdn: int, mode: str):
    """
    Mutates ONLY the committed recurrent S for one GDN.

    target_gdn:
        4 -> GDN1
        5 -> GDN2

    mode:
        zero    -> replace committed S with exact zeros
        shuffle -> batch-permute committed S, preserving its values
    """
    target = state.ctx.gdn[target_gdn]

    if target.committed_state is None:
        raise RuntimeError(
            f"GDN{target_gdn - 3} committed_state is unexpectedly None."
        )

    before_state = target.committed_state.detach().clone()
    before_conv = clone_tensor(target.committed_conv_buffer)

    # Non-target GDN state controls.
    other_gdn = 5 if target_gdn == 4 else 4
    before_other_state = clone_tensor(
        state.ctx.gdn[other_gdn].committed_state
    )
    before_other_conv = clone_tensor(
        state.ctx.gdn[other_gdn].committed_conv_buffer
    )

    if mode == "zero":
        target.committed_state.zero_()

    elif mode == "shuffle":
        B = target.committed_state.shape[0]

        if B < 2:
            raise RuntimeError("Shuffle intervention requires batch size >= 2.")

        perm = torch.randperm(B, device=target.committed_state.device)

        # Avoid the identity permutation so the intervention is genuine.
        if torch.equal(perm, torch.arange(B, device=perm.device)):
            perm = torch.roll(perm, shifts=1, dims=0)

        target.committed_state.copy_(
            target.committed_state.index_select(0, perm)
        )

    else:
        raise ValueError(mode)

    # Verification: target changed, non-target pathways did not.
    after_state = target.committed_state.detach()

    if mode == "zero":
        if not torch.equal(after_state, torch.zeros_like(after_state)):
            raise AssertionError("ZERO intervention failed.")

    if torch.equal(before_state, after_state):
        raise AssertionError(
            f"{mode.upper()} intervention produced no change in target GDN "
            "committed state."
        )

    if before_conv is not None and not torch.equal(
        before_conv,
        target.committed_conv_buffer,
    ):
        raise AssertionError(
            "Intervention modified target GDN convolution buffer."
        )

    if before_other_state is not None and not torch.equal(
        before_other_state,
        state.ctx.gdn[other_gdn].committed_state,
    ):
        raise AssertionError(
            "Intervention modified non-target GDN committed state."
        )

    if before_other_conv is not None and not torch.equal(
        before_other_conv,
        state.ctx.gdn[other_gdn].committed_conv_buffer,
    ):
        raise AssertionError(
            "Intervention modified non-target GDN convolution buffer."
        )

    return {
        "target_shape": list(before_state.shape),
        "target_numel_per_example": int(before_state[0].numel()),
        "target_pre_norm": float(
            before_state.float().reshape(before_state.shape[0], -1).norm(dim=1).mean()
        ),
        "target_post_norm": float(
            after_state.float().reshape(after_state.shape[0], -1).norm(dim=1).mean()
        ),
        "conv_unchanged": True,
        "other_gdn_state_unchanged": True,
        "other_gdn_conv_unchanged": True,
    }


@torch.no_grad()
def run_condition(bits, target_gdn=None, mode=None):
    model = load_model()

    x = task.token_features[bits]

    B = bits.shape[0]
    state = model.initial_persistent_state(
        batch_size=B,
        device=DEVICE,
        dtype=x.dtype,
    )

    C_prev = torch.zeros(
        B,
        DIM,
        device=DEVICE,
        dtype=x.dtype,
    )

    intervention_records = []

    # The committed S produced at token t is the recurrent state available
    # to token t+1. Therefore intervention occurs between steps.
    for t in range(bits.shape[1]):
        out = model.step(
            x[:, t, :],
            state,
            C_prev,
        )

        state = out["state"]

        # No intervention after the final token because it cannot affect the
        # final-token output. This keeps the causal intervention well-defined.
        if (
            target_gdn is not None
            and mode is not None
            and t < bits.shape[1] - 1
        ):
            rec = intervene_state(
                state,
                target_gdn=target_gdn,
                mode=mode,
            )
            rec["token"] = t + 1
            intervention_records.append(rec)

    logits = out["logits"]
    target = bits.sum(dim=1) % 2

    ce = F.cross_entropy(logits, target).item()
    pred = logits.argmax(dim=-1)
    acc = (pred == target).float().mean().item()

    return {
        "loss": ce,
        "accuracy": acc,
        "intervention_records": intervention_records,
        "final_logit_shape": list(logits.shape),
    }


def summarize_records(records):
    if not records:
        return {}

    first = records[0]
    return {
        "num_interventions": len(records),
        "target_shape": first["target_shape"],
        "target_numel_per_example": first["target_numel_per_example"],
        "pre_norm_mean": sum(r["target_pre_norm"] for r in records) / len(records),
        "post_norm_mean": sum(r["target_post_norm"] for r in records) / len(records),
        "all_conv_unchanged": all(
            r["conv_unchanged"] for r in records
        ),
        "all_other_gdn_state_unchanged": all(
            r["other_gdn_state_unchanged"] for r in records
        ),
        "all_other_gdn_conv_unchanged": all(
            r["other_gdn_conv_unchanged"] for r in records
        ),
    }


task = TaskAParity(dim=DIM, device=DEVICE)

print("=" * 96)
print("GATE A4 — TRUE PERSISTENT GDN STATE ABLATION")
print("=" * 96)
print(f"device={DEVICE}")
print(f"dim={DIM}")
print(f"checkpoint={CKPT}")
print("intervention target = state.ctx.gdn[i].committed_state")
print("intervention timing = after token t, before token t+1")
print("target recurrent convolution buffer = PRESERVED")
print("non-target GDN state = PRESERVED")
print("final-token intervention = OMITTED")
print("conditions = CLEAN, GDN1_ZERO, GDN2_ZERO, GDN1_SHUFFLE, GDN2_SHUFFLE")
print("lengths = L8, L128")
print()

results = {}

for length in (8, 128):
    bits = make_bits(length)

    print("-" * 96)
    print(f"LENGTH {length}  examples={bits.shape[0]}")
    print("-" * 96)

    clean = run_condition(bits)
    results[f"L{length}_clean"] = clean

    print(
        f"CLEAN          CE={clean['loss']:.6f} "
        f"ACC={clean['accuracy']:.4f}"
    )

    for target_gdn, gdn_name in ((4, "GDN1"), (5, "GDN2")):
        for mode, mode_name in (("zero", "ZERO"), ("shuffle", "SHUFFLE")):
            result = run_condition(
                bits,
                target_gdn=target_gdn,
                mode=mode,
            )

            key = f"L{length}_{gdn_name}_{mode_name}"
            results[key] = result

            control = summarize_records(result["intervention_records"])

            print(
                f"{gdn_name + '_' + mode_name:<14} "
                f"CE={result['loss']:.6f} "
                f"ACC={result['accuracy']:.4f} "
                f"ΔACC={result['accuracy'] - clean['accuracy']:+.4f} "
                f"interventions={control['num_interventions']} "
                f"shape={control['target_shape']} "
                f"other_state_ok={control['all_other_gdn_state_unchanged']} "
                f"conv_ok={control['all_conv_unchanged']}"
            )

            if not control["all_other_gdn_state_unchanged"]:
                raise AssertionError("Non-target GDN state changed.")

            if not control["all_other_gdn_conv_unchanged"]:
                raise AssertionError("Non-target GDN convolution changed.")

            if not control["all_conv_unchanged"]:
                raise AssertionError("Target GDN convolution buffer changed.")


out = {
    "gate": "A4",
    "checkpoint": CKPT,
    "device": str(DEVICE),
    "dim": DIM,
    "target": "persistent committed GDN recurrent state S",
    "target_paths": {
        "GDN1": "state.ctx.gdn[4].committed_state",
        "GDN2": "state.ctx.gdn[5].committed_state",
    },
    "timing": "after token t before token t+1",
    "conditions": [
        "clean",
        "GDN1_ZERO",
        "GDN2_ZERO",
        "GDN1_SHUFFLE",
        "GDN2_SHUFFLE",
    ],
    "results": {
        key: {
            "loss": value["loss"],
            "accuracy": value["accuracy"],
            "final_logit_shape": value["final_logit_shape"],
            "intervention_summary": summarize_records(
                value["intervention_records"]
            ),
        }
        for key, value in results.items()
    },
}

Path("experiments/gateA_a4_state_ablation.json").write_text(
    json.dumps(out, indent=2)
)

print("\n" + "=" * 96)
print("=== GATE A4 COMPLETE ===")
print("WROTE experiments/gateA_a4_state_ablation.json")
print("=" * 96)
