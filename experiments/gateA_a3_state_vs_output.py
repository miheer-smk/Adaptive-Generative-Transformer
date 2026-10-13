from __future__ import annotations

import itertools
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agt import AGTv3
from data.task_a import TaskAParity


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIM = 64
CKPT = "experiments/runs/stage1_parity_seed3305_best.pt"

TRAIN_SEED = 4401
EVAL_SEEDS = {
    8: 8008,
    16: 8016,
    32: 8032,
    64: 8064,
    128: 8128,
}


def make_model(trained: bool) -> AGTv3:
    if trained:
        # Exact successful checkpoint; no retraining.
        model = AGTv3(
            dim=DIM,
            num_memory_slots=16,
            vocab_size=2,
        ).to(DEVICE)

        checkpoint = torch.load(CKPT, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    # Exact control seed used for the earlier direct-state probe.
    torch.manual_seed(TRAIN_SEED)

    model = AGTv3(
        dim=DIM,
        num_memory_slots=16,
        vocab_size=2,
    ).to(DEVICE)

    model.eval()
    return model


def exhaustive_train_batches(device):
    batches = []

    for length in range(4, 9):
        bits = torch.tensor(
            list(itertools.product([0, 1], repeat=length)),
            dtype=torch.long,
            device=device,
        )

        batches.append(bits)

    return batches


def make_eval_bits(length: int, device):
    if length == 8:
        return torch.tensor(
            list(itertools.product([0, 1], repeat=8)),
            dtype=torch.long,
            device=device,
        )

    n = 256 if length == 16 else 512

    g = torch.Generator()
    g.manual_seed(EVAL_SEEDS[length])

    return torch.randint(
        0,
        2,
        (n, length),
        generator=g,
        dtype=torch.long,
        device="cpu",
    ).to(device)


@torch.no_grad()
def collect_representations(model: AGTv3, bit_batches):
    """
    Collect:

      GDN1_OUTPUT = AGTBlock block-4 output[0]:
                     post-MHC residual h after GDN1.

      GDN2_OUTPUT = AGTBlock block-5 output[0]:
                     post-MHC residual h after GDN2.

      GDN1_STATE = persistent committed S after token.

      GDN2_STATE = persistent committed S after token.

    The hooks only copy outputs; they do not replace or alter them.
    """
    captures = {
        4: None,
        5: None,
    }

    def make_hook(block_id):
        def hook(module, inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError(
                    f"GDN block {block_id} returned unexpected output."
                )

            y, ephemeral_state = output

            captures[block_id] = y.detach().clone()

            if ephemeral_state.ndim != 4:
                raise RuntimeError(
                    f"GDN block {block_id} state has unexpected shape "
                    f"{tuple(ephemeral_state.shape)}"
                )

        return hook

    h4 = model.blocks.blocks[4].register_forward_hook(make_hook(4))
    h5 = model.blocks.blocks[5].register_forward_hook(make_hook(5))

    out_data = {
        "gdn1_output": [],
        "gdn2_output": [],
        "gdn1_state": [],
        "gdn2_state": [],
        "target": [],
    }

    try:
        for bits in bit_batches:
            x = task.token_features[bits]

            state = model.initial_persistent_state(
                batch_size=bits.shape[0],
                device=DEVICE,
                dtype=x.dtype,
            )

            C_prev = torch.zeros(
                bits.shape[0],
                DIM,
                device=DEVICE,
                dtype=x.dtype,
            )

            # Collect only the FINAL token representation for each sequence.
            # This exactly matches the final-state parity classification setup.
            for t in range(bits.shape[1]):
                captures[4] = None
                captures[5] = None

                out = model.step(
                    x[:, t, :],
                    state,
                    C_prev,
                )

                state = out["state"]

            if captures[4] is None or captures[5] is None:
                raise RuntimeError("GDN block hooks did not fire.")

            S1 = state.ctx.gdn[4].committed_state
            S2 = state.ctx.gdn[5].committed_state

            if S1 is None or S2 is None:
                raise RuntimeError("Missing committed GDN state.")

            out_data["gdn1_output"].append(
                captures[4].reshape(bits.shape[0], -1).float()
            )
            out_data["gdn2_output"].append(
                captures[5].reshape(bits.shape[0], -1).float()
            )
            out_data["gdn1_state"].append(
                S1.reshape(bits.shape[0], -1).float()
            )
            out_data["gdn2_state"].append(
                S2.reshape(bits.shape[0], -1).float()
            )
            out_data["target"].append(
                (bits.sum(dim=1) % 2).long()
            )

    finally:
        h4.remove()
        h5.remove()

    for key in (
        "gdn1_output",
        "gdn2_output",
        "gdn1_state",
        "gdn2_state",
        "target",
    ):
        out_data[key] = torch.cat(out_data[key], dim=0)

    return out_data


def train_probe(X, y, steps=500):
    """
    Independent linear probe.

    Same protocol as the previous direct-state study:
      - feature standardization from probe-training set
      - linear 2-class head
      - Adam
      - lr=3e-3
      - full-batch optimization
    """
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(1e-5)

    Xn = (X - mean) / std

    probe = nn.Linear(X.shape[1], 2).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=3e-3)

    for _ in range(steps):
        logits = probe(Xn)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return probe, mean, std


def evaluate_probe(probe, mean, std, X, y):
    Xn = (X - mean) / std

    with torch.no_grad():
        logits = probe(Xn)
        loss = F.cross_entropy(logits, y).item()
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()

    return acc, loss


def run_suite(trained: bool):
    model = make_model(trained)

    train_data = collect_representations(
        model,
        exhaustive_train_batches(DEVICE),
    )

    probe_info = {}

    for key in (
        "gdn1_output",
        "gdn2_output",
        "gdn1_state",
        "gdn2_state",
    ):
        probe, mean, std = train_probe(
            train_data[key],
            train_data["target"],
        )

        train_acc, train_ce = evaluate_probe(
            probe,
            mean,
            std,
            train_data[key],
            train_data["target"],
        )

        probe_info[key] = {
            "probe": probe,
            "mean": mean,
            "std": std,
            "train_acc": train_acc,
            "train_ce": train_ce,
            "dim": int(train_data[key].shape[1]),
        }

    results = []

    for length in (8, 16, 32, 64, 128):
        bits = make_eval_bits(length, DEVICE)

        eval_data = collect_representations(
            model,
            [bits],
        )

        row = {
            "length": length,
            "examples": int(bits.shape[0]),
        }

        for key in (
            "gdn1_output",
            "gdn2_output",
            "gdn1_state",
            "gdn2_state",
        ):
            info = probe_info[key]

            acc, ce = evaluate_probe(
                info["probe"],
                info["mean"],
                info["std"],
                eval_data[key],
                eval_data["target"],
            )

            row[f"{key}_acc"] = acc
            row[f"{key}_ce"] = ce

        results.append(row)

    return probe_info, results


task = TaskAParity(dim=DIM, device=DEVICE)

torch.manual_seed(TRAIN_SEED)

print("=" * 88)
print("GATE A3 — GDN RECURRENT STATE vs GDN OUTPUT REPRESENTATION PROBES")
print("=" * 88)
print(f"device={DEVICE}")
print(f"dim={DIM}")
print(f"trained_checkpoint={CKPT}")
print("training representation lengths = exhaustive L4..L8")
print("evaluation lengths = L8, L16, L32, L64, L128")
print("probe = standardized linear 2-class probe, Adam lr=3e-3, 500 steps")
print()
print("GDN output definition:")
print("  AGTBlock block output[0] = post-MHC residual h")
print("  This is the same tensor targeted by the prior GDN-output ablation.")
print()
print("GDN state definition:")
print("  state.ctx.gdn[i].committed_state")
print("  This is persistent recurrent S crossing token boundaries.")
print("  D64 state shape = [B, 6, 128, 64] = 49,152 scalars/example.")
print()


trained_probe_info, trained_results = run_suite(True)

print("\n" + "-" * 88)
print("TRAINED MODEL — PROBE TRAINING")
print("-" * 88)

for key, info in trained_probe_info.items():
    print(
        f"{key:<16} "
        f"dim={info['dim']:<6} "
        f"train_acc={info['train_acc']:.4f} "
        f"train_ce={info['train_ce']:.6f}"
    )

print("\n" + "-" * 88)
print("TRAINED MODEL — ZERO-/HELD-OUT LENGTH EVALUATION")
print("-" * 88)

for row in trained_results:
    print(
        f"L={row['length']:<3} "
        f"GDN1_OUT={row['gdn1_output_acc']:.4f} "
        f"GDN2_OUT={row['gdn2_output_acc']:.4f} "
        f"GDN1_STATE={row['gdn1_state_acc']:.4f} "
        f"GDN2_STATE={row['gdn2_state_acc']:.4f}"
    )

untrained_probe_info, untrained_results = run_suite(False)

print("\n" + "-" * 88)
print("UNTRAINED CONTROL — PROBE TRAINING")
print("-" * 88)

for key, info in untrained_probe_info.items():
    print(
        f"{key:<16} "
        f"dim={info['dim']:<6} "
        f"train_acc={info['train_acc']:.4f} "
        f"train_ce={info['train_ce']:.6f}"
    )

print("\n" + "-" * 88)
print("UNTRAINED CONTROL — LENGTH EVALUATION")
print("-" * 88)

for row in untrained_results:
    print(
        f"L={row['length']:<3} "
        f"GDN1_OUT={row['gdn1_output_acc']:.4f} "
        f"GDN2_OUT={row['gdn2_output_acc']:.4f} "
        f"GDN1_STATE={row['gdn1_state_acc']:.4f} "
        f"GDN2_STATE={row['gdn2_state_acc']:.4f}"
    )


report = {
    "gate": "A3",
    "device": str(DEVICE),
    "dim": DIM,
    "checkpoint": CKPT,
    "trained_probe_training": {
        key: {
            "dim": info["dim"],
            "train_acc": info["train_acc"],
            "train_ce": info["train_ce"],
        }
        for key, info in trained_probe_info.items()
    },
    "trained_evaluation": trained_results,
    "untrained_probe_training": {
        key: {
            "dim": info["dim"],
            "train_acc": info["train_acc"],
            "train_ce": info["train_ce"],
        }
        for key, info in untrained_probe_info.items()
    },
    "untrained_evaluation": untrained_results,
    "protocol": {
        "train_lengths": [4, 5, 6, 7, 8],
        "eval_lengths": [8, 16, 32, 64, 128],
        "feature_seed": task.FEATURE_SEED,
        "task_seed": task.TASK_SEED,
        "probe_seed": TRAIN_SEED,
        "probe_steps": 500,
        "probe_lr": 3e-3,
    },
}

Path("experiments/gateA_a3_state_vs_output.json").write_text(
    json.dumps(report, indent=2)
)

print("\n" + "=" * 88)
print("=== GATE A3 COMPLETE ===")
print("WROTE experiments/gateA_a3_state_vs_output.json")
print("=" * 88)
