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

PREFIX_LEN = 8
CONTINUATION = 0
BATCH_DTYPE = torch.float32


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


def make_opposite_parity_pairs():
    """
    128 pairs.

    donor      = [0] + suffix
    recipient  = [1] + suffix

    Same suffix => matched continuation.
    First bit flip => opposite prefix parity.
    """
    pairs = []

    for suffix in itertools.product([0, 1], repeat=7):
        donor = (0,) + suffix
        recipient = (1,) + suffix

        pairs.append((donor, recipient))

    return pairs


def make_same_parity_pairs():
    """
    64 same-parity donor/recipient controls.

    donor      = [0, 0] + suffix
    recipient  = [1, 1] + suffix

    Two flipped bits preserve parity while changing the prefix.
    """
    pairs = []

    for suffix in itertools.product([0, 1], repeat=6):
        donor = (0, 0) + suffix
        recipient = (1, 1) + suffix

        assert (
            sum(donor) % 2
            == sum(recipient) % 2
        )

        pairs.append((donor, recipient))

    return pairs


def to_tensor(sequences):
    return torch.tensor(
        sequences,
        dtype=torch.long,
        device=DEVICE,
    )


@torch.no_grad()
def run_prefix(model, bits):
    """
    Process the matched 8-token prefix and return persistent states.

    The returned state is H_8, so its committed GDN state S_8 is exactly
    the recurrent state read when processing token 9.
    """
    x = task.token_features[bits]

    B = bits.shape[0]

    state = model.initial_persistent_state(
        batch_size=B,
        device=DEVICE,
        dtype=x.dtype,
    )

    # Validated Stage-1 protocol: explicit compatibility input, unchanged.
    C_prev = torch.zeros(
        B,
        model.dim,
        device=DEVICE,
        dtype=x.dtype,
    )

    for t in range(PREFIX_LEN):
        out = model.step(
            x[:, t, :],
            state,
            C_prev,
        )
        state = out["state"]

    if state.tokens_committed != PREFIX_LEN:
        raise AssertionError(
            f"Expected tokens_committed={PREFIX_LEN}, "
            f"got {state.tokens_committed}"
        )

    return state


def patch_states(
    recipient_state,
    donor_state,
    targets,
):
    """
    Copy only selected GDN committed S from donor -> recipient.

    targets:
      ("GDN1",)
      ("GDN2",)
      ("GDN1", "GDN2")
    """
    patched = recipient_state.clone()

    snapshots = {}

    index_map = {
        "GDN1": 4,
        "GDN2": 5,
    }

    for name in targets:
        idx = index_map[name]

        rec_cache = patched.ctx.gdn[idx]
        donor_cache = donor_state.ctx.gdn[idx]

        if rec_cache.committed_state is None:
            raise RuntimeError(f"{name} recipient state is None.")
        if donor_cache.committed_state is None:
            raise RuntimeError(f"{name} donor state is None.")

        # Snapshot all relevant context before patch.
        snapshots[name] = {
            "recipient_state_before": rec_cache.committed_state.clone(),
            "recipient_conv_before": (
                rec_cache.committed_conv_buffer.clone()
                if rec_cache.committed_conv_buffer is not None
                else None
            ),
        }

        rec_cache.committed_state.copy_(
            donor_cache.committed_state
        )

        if torch.equal(
            snapshots[name]["recipient_state_before"],
            rec_cache.committed_state,
        ):
            raise AssertionError(
                f"{name} patch did not change recipient committed state."
            )

        # The recurrent convolution buffer MUST remain recipient-owned.
        before_conv = snapshots[name]["recipient_conv_before"]

        if before_conv is not None and not torch.equal(
            before_conv,
            rec_cache.committed_conv_buffer,
        ):
            raise AssertionError(
                f"{name} patch changed recipient convolution buffer."
            )

    return patched, snapshots


@torch.no_grad()
def continue_one_token(model, state):
    """
    Process the single common continuation token 0.

    Since token 0 does not change parity, donor/recipient labels remain
    opposite after continuation.
    """
    B = state.s.shape[0]

    x = task.token_features[
        torch.full(
            (B, 1),
            CONTINUATION,
            dtype=torch.long,
            device=DEVICE,
        )
    ][:, 0, :]

    C_prev = torch.zeros(
        B,
        model.dim,
        device=DEVICE,
        dtype=x.dtype,
    )

    out = model.step(
        x,
        state,
        C_prev,
    )

    return out


def parity_metrics(logits, donor_labels, recipient_labels):
    probs = torch.softmax(logits, dim=-1)

    donor_p = probs.gather(
        1,
        donor_labels[:, None],
    ).squeeze(1)

    recipient_p = probs.gather(
        1,
        recipient_labels[:, None],
    ).squeeze(1)

    donor_logit = logits.gather(
        1,
        donor_labels[:, None],
    ).squeeze(1)

    recipient_logit = logits.gather(
        1,
        recipient_labels[:, None],
    ).squeeze(1)

    pred = logits.argmax(dim=-1)

    return {
        "mean_p_donor": float(donor_p.mean()),
        "mean_p_recipient": float(recipient_p.mean()),
        "mean_donor_margin": float(
            (donor_logit - recipient_logit).mean()
        ),
        "donor_argmax_rate": float(
            (pred == donor_labels).float().mean()
        ),
        "recipient_argmax_rate": float(
            (pred == recipient_labels).float().mean()
        ),
        "mean_cross_entropy_to_recipient": float(
            F.cross_entropy(logits, recipient_labels)
        ),
        "mean_cross_entropy_to_donor": float(
            F.cross_entropy(logits, donor_labels)
        ),
    }


def run_opposite_parity_suite(model):
    pairs = make_opposite_parity_pairs()

    donor_bits = to_tensor([p[0] for p in pairs])
    recipient_bits = to_tensor([p[1] for p in pairs])

    donor_labels = donor_bits.sum(dim=1) % 2
    recipient_labels = recipient_bits.sum(dim=1) % 2

    assert torch.all(donor_labels != recipient_labels)

    donor_state = run_prefix(model, donor_bits)
    recipient_state = run_prefix(model, recipient_bits)

    clean_donor = continue_one_token(model, donor_state)
    clean_recipient = continue_one_token(
        model,
        recipient_state.clone(),
    )

    clean_donor_metrics = parity_metrics(
        clean_donor["logits"],
        donor_labels,
        recipient_labels,
    )

    clean_recipient_metrics = parity_metrics(
        clean_recipient["logits"],
        donor_labels,
        recipient_labels,
    )

    # Conditions to test.
    patch_conditions = {
        "GDN1": ("GDN1",),
        "GDN2": ("GDN2",),
        "BOTH": ("GDN1", "GDN2"),
    }

    outputs = {
        "clean_donor": clean_donor_metrics,
        "clean_recipient": clean_recipient_metrics,
    }

    for condition, targets in patch_conditions.items():
        patched_recipient, snapshots = patch_states(
            recipient_state,
            donor_state,
            targets,
        )

        out = continue_one_token(
            model,
            patched_recipient,
        )

        metrics = parity_metrics(
            out["logits"],
            donor_labels,
            recipient_labels,
        )

        clean_p_donor = torch.softmax(
            clean_recipient["logits"],
            dim=-1,
        ).gather(
            1,
            donor_labels[:, None],
        ).squeeze(1)

        patched_p_donor = torch.softmax(
            out["logits"],
            dim=-1,
        ).gather(
            1,
            donor_labels[:, None],
        ).squeeze(1)

        clean_margin = (
            clean_recipient["logits"].gather(
                1,
                donor_labels[:, None],
            ).squeeze(1)
            -
            clean_recipient["logits"].gather(
                1,
                recipient_labels[:, None],
            ).squeeze(1)
        )

        patched_margin = (
            out["logits"].gather(
                1,
                donor_labels[:, None],
            ).squeeze(1)
            -
            out["logits"].gather(
                1,
                recipient_labels[:, None],
            ).squeeze(1)
        )

        metrics["delta_p_donor"] = float(
            (patched_p_donor - clean_p_donor).mean()
        )

        metrics["delta_donor_margin"] = float(
            (patched_margin - clean_margin).mean()
        )

        metrics["pairwise_donor_probability_increase_rate"] = float(
            (patched_p_donor > clean_p_donor).float().mean()
        )

        metrics["patch_flip_to_donor_rate"] = float(
            out["logits"].argmax(dim=-1).eq(donor_labels).float().mean()
        )

        outputs[condition] = metrics

        # Verify the unselected GDN stays recipient-owned.
        selected = set(targets)
        for name, idx in (("GDN1", 4), ("GDN2", 5)):
            cache = patched_recipient.ctx.gdn[idx]

            if name not in selected:
                original = (
                    recipient_state.ctx.gdn[idx].committed_state
                )
                if not torch.equal(
                    original,
                    cache.committed_state,
                ):
                    raise AssertionError(
                        f"Unselected {name} state changed."
                    )

    outputs["num_pairs"] = len(pairs)
    outputs["prefix_len"] = PREFIX_LEN
    outputs["continuation"] = CONTINUATION

    return outputs


def run_same_parity_controls(model):
    pairs = make_same_parity_pairs()

    donor_bits = to_tensor([p[0] for p in pairs])
    recipient_bits = to_tensor([p[1] for p in pairs])

    donor_labels = donor_bits.sum(dim=1) % 2
    recipient_labels = recipient_bits.sum(dim=1) % 2

    assert torch.all(donor_labels == recipient_labels)

    donor_state = run_prefix(model, donor_bits)
    recipient_state = run_prefix(model, recipient_bits)

    clean_recipient = continue_one_token(
        model,
        recipient_state.clone(),
    )

    clean_logits = clean_recipient["logits"]

    outputs = {
        "num_pairs": len(pairs),
        "prefix_len": PREFIX_LEN,
        "continuation": CONTINUATION,
    }

    for condition, targets in {
        "GDN1": ("GDN1",),
        "GDN2": ("GDN2",),
        "BOTH": ("GDN1", "GDN2"),
    }.items():
        patched_recipient, _ = patch_states(
            recipient_state,
            donor_state,
            targets,
        )

        out = continue_one_token(
            model,
            patched_recipient,
        )

        clean_prob = torch.softmax(
            clean_logits,
            dim=-1,
        )

        patched_prob = torch.softmax(
            out["logits"],
            dim=-1,
        )

        correct = recipient_labels[:, None]

        clean_p = clean_prob.gather(
            1,
            correct,
        ).squeeze(1)

        patched_p = patched_prob.gather(
            1,
            correct,
        ).squeeze(1)

        clean_pred = clean_logits.argmax(dim=-1)
        patched_pred = out["logits"].argmax(dim=-1)

        outputs[condition] = {
            "mean_p_correct_clean": float(clean_p.mean()),
            "mean_p_correct_patched": float(patched_p.mean()),
            "delta_p_correct": float(
                (patched_p - clean_p).mean()
            ),
            "patched_correct_argmax_rate": float(
                (patched_pred == recipient_labels).float().mean()
            ),
            "patched_flip_rate": float(
                (patched_pred != recipient_labels).float().mean()
            ),
        }

    return outputs


task = TaskAParity(dim=DIM, device=DEVICE)
model = load_model()

print("=" * 100)
print("GATE A6 — CAUSAL GDN STATE PATCHING")
print("=" * 100)
print(f"device={DEVICE}")
print(f"dim={DIM}")
print(f"checkpoint={CKPT}")
print(f"prefix_len={PREFIX_LEN}")
print(f"common_continuation_token={CONTINUATION}")
print()
print("PATCH POINT:")
print("  after prefix H_8")
print("  before token 9")
print("  target = committed GDN recurrent state S_8")
print()
print("OPPOSITE-PARITY PAIRS:")
print("  donor      = [0] + suffix")
print("  recipient  = [1] + suffix")
print("  same suffix / opposite parity")
print("  both donor->recipient and reverse direction")
print()
print("SAME-PARITY CONTROLS:")
print("  donor      = [0,0] + suffix")
print("  recipient  = [1,1] + suffix")
print("  same parity / different prefix")
print()
print("PATCH CONDITIONS:")
print("  GDN1, GDN2, BOTH")
print("  only committed_state copied")
print("  convolution buffer and all other persistent context preserved")
print()

opposite = run_opposite_parity_suite(model)
same = run_same_parity_controls(model)

print("-" * 100)
print("OPPOSITE-PARITY RESULTS — FORWARD DIRECTION")
print("-" * 100)

for key in (
    "clean_donor",
    "clean_recipient",
    "GDN1",
    "GDN2",
    "BOTH",
):
    r = opposite[key]

    print(
        f"{key:<16} "
        f"P(donor)={r['mean_p_donor']:.4f} "
        f"P(recipient)={r['mean_p_recipient']:.4f} "
        f"donor_margin={r['mean_donor_margin']:+.4f} "
        f"donor_argmax={r['donor_argmax_rate']:.4f}"
    )

    if key in ("GDN1", "GDN2", "BOTH"):
        print(
            f"{'':16} "
            f"ΔP(donor)={r['delta_p_donor']:+.4f} "
            f"Δmargin={r['delta_donor_margin']:+.4f} "
            f"P(donor) increase={r['pairwise_donor_probability_increase_rate']:.4f} "
            f"flip_to_donor={r['patch_flip_to_donor_rate']:.4f}"
        )

print()
print("-" * 100)
print("SAME-PARITY DONOR CONTROLS")
print("-" * 100)

for key in ("GDN1", "GDN2", "BOTH"):
    r = same[key]
    print(
        f"{key:<16} "
        f"P(correct) clean={r['mean_p_correct_clean']:.4f} "
        f"patched={r['mean_p_correct_patched']:.4f} "
        f"Δ={r['delta_p_correct']:+.4f} "
        f"patched_correct={r['patched_correct_argmax_rate']:.4f} "
        f"flip={r['patched_flip_rate']:.4f}"
    )

report = {
    "gate": "A6",
    "checkpoint": CKPT,
    "device": str(DEVICE),
    "dim": DIM,
    "prefix_len": PREFIX_LEN,
    "continuation_token": CONTINUATION,
    "patch_point": "H_8 -> before token 9",
    "target_paths": {
        "GDN1": "state.ctx.gdn[4].committed_state",
        "GDN2": "state.ctx.gdn[5].committed_state",
    },
    "patch_definition": (
        "copy donor committed_state into recipient committed_state; "
        "leave all other persistent context unchanged"
    ),
    "opposite_parity": opposite,
    "same_parity_controls": same,
}

Path("experiments/gateA_a6_state_patching.json").write_text(
    json.dumps(report, indent=2)
)

print()
print("=" * 100)
print("=== GATE A6 COMPLETE ===")
print("WROTE experiments/gateA_a6_state_patching.json")
print("=" * 100)
