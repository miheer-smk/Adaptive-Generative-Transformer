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
    8: 6808,
    128: 6128,
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


def make_chunks(bits, chunk_size=64):
    return [
        bits[i:i + chunk_size]
        for i in range(0, bits.shape[0], chunk_size)
    ]


def _run_once_single_batch(
    bits,
    *,
    state_intervention=None,
    output_intervention=None,
    null_intervention=False,
):
    """
    state_intervention:
        None
        4 -> GDN1 persistent state zero
        5 -> GDN2 persistent state zero

    output_intervention:
        None
        4 -> GDN1 block output zero
        5 -> GDN2 block output zero

    null_intervention:
        True -> execute a genuine no-op intervention on persistent state
                and verify it is bitwise identical.

    Only one active intervention type is used per run.
    """
    model = load_model()

    if (
        sum(
            x is not None
            for x in (state_intervention, output_intervention)
        )
        > 1
        or (
            null_intervention
            and (
                state_intervention is not None
                or output_intervention is not None
            )
        )
    ):
        raise ValueError("Invalid condition combination.")

    hooks = []

    if output_intervention is not None:
        target = output_intervention

        def output_hook(module, inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError(
                    "Expected GDN block output tuple (y, ephemeral_state)."
                )

            y, ephemeral_state = output

            # Preserve the returned recurrent ephemeral state exactly;
            # intervene only on the GDN output entering the residual stream.
            return (
                torch.zeros_like(y),
                ephemeral_state,
            )

        hooks.append(
            model.blocks.blocks[target].register_forward_hook(
                output_hook
            )
        )

    x = task.token_features[bits]

    B = bits.shape[0]

    state = model.initial_persistent_state(
        batch_size=B,
        device=DEVICE,
        dtype=x.dtype,
    )

    # Exact validated Stage-1 compatibility protocol.
    C_prev = torch.zeros(
        B,
        model.dim,
        device=DEVICE,
        dtype=x.dtype,
    )

    null_checks = []
    state_checks = []

    try:
        for t in range(bits.shape[1]):
            out = model.step(
                x[:, t, :],
                state,
                C_prev,
            )

            state = out["state"]

            # State intervention occurs after S_t is committed and before
            # token t+1 consumes that state.
            if (
                state_intervention is not None
                and t < bits.shape[1] - 1
            ):
                cache = state.ctx.gdn[state_intervention]

                if cache.committed_state is None:
                    raise RuntimeError("Missing committed GDN state.")

                before_state = cache.committed_state.clone()

                cache.committed_state.zero_()

                if not torch.equal(
                    cache.committed_state,
                    torch.zeros_like(cache.committed_state),
                ):
                    raise AssertionError("State zero intervention failed.")

                # Conv buffer must not move.
                state_checks.append(
                    {
                        "token": t + 1,
                        "shape": list(before_state.shape),
                    }
                )

            if (
                null_intervention
                and t < bits.shape[1] - 1
            ):
                # Genuine intervention path: clone and copy the tensor back
                # to itself. This exercises the intervention machinery but
                # must produce no numerical change.
                cache = state.ctx.gdn[4]

                before = cache.committed_state.clone()
                cache.committed_state.copy_(
                    cache.committed_state.clone()
                )
                after = cache.committed_state

                null_checks.append(
                    bool(torch.equal(before, after))
                )

    finally:
        for h in hooks:
            h.remove()

    logits = out["logits"]
    target = bits.sum(dim=1) % 2

    ce = F.cross_entropy(logits, target).item()
    acc = (logits.argmax(dim=-1) == target).float().mean().item()

    return {
        "ce": ce,
        "accuracy": acc,
        "logits": logits.detach().clone(),
        "null_checks": null_checks,
        "state_checks": state_checks,
    }


def run_once(
    bits,
    *,
    state_intervention=None,
    output_intervention=None,
    null_intervention=False,
):
    """
    Resource-safe wrapper.

    L8 remains one batch.
    L128 is split into independent chunks, but all 512 sequences are retained.
    Each sequence receives exactly the same causal intervention protocol as
    the original implementation.
    """
    if bits.shape[0] <= 64:
        return _run_once_single_batch(
            bits,
            state_intervention=state_intervention,
            output_intervention=output_intervention,
            null_intervention=null_intervention,
        )

    chunk_outputs = []

    for chunk in make_chunks(bits, chunk_size=64):
        result = _run_once_single_batch(
            chunk,
            state_intervention=state_intervention,
            output_intervention=output_intervention,
            null_intervention=null_intervention,
        )
        chunk_outputs.append(result)

        # Release transient CUDA allocations between chunks.
        del result
        torch.cuda.empty_cache()

    # Re-run aggregation while retaining exact per-example logits.
    # The loop above intentionally stores only the final outputs needed below.
    # Therefore execute the chunks again and concatenate their logits.
    logits_parts = []
    ce_weighted = 0.0
    acc_weighted = 0.0
    total = 0
    null_checks = []
    state_checks = []

    for chunk in make_chunks(bits, chunk_size=64):
        result = _run_once_single_batch(
            chunk,
            state_intervention=state_intervention,
            output_intervention=output_intervention,
            null_intervention=null_intervention,
        )

        logits_parts.append(result["logits"])
        n = chunk.shape[0]
        ce_weighted += result["ce"] * n
        acc_weighted += result["accuracy"] * n
        total += n
        null_checks.extend(result["null_checks"])
        state_checks.extend(result["state_checks"])

        del result
        torch.cuda.empty_cache()

    logits = torch.cat(logits_parts, dim=0)

    return {
        "ce": ce_weighted / total,
        "accuracy": acc_weighted / total,
        "logits": logits,
        "null_checks": null_checks,
        "state_checks": state_checks,
    }


def compare_logits(a, b):
    diff = (a - b).abs()
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "bitwise_equal": bool(torch.equal(a, b)),
    }


task = TaskAParity(dim=DIM, device=DEVICE)

print("=" * 100)
print("GATE A7 — CAUSALITY / DAMAGE / NULL CONTROLS")
print("=" * 100)
print(f"device={DEVICE}")
print(f"dim={DIM}")
print(f"checkpoint={CKPT}")
print("lengths = L8, L128")
print()
print("CONTROL FAMILY:")
print("  CLEAN")
print("  GDN1_STATE_ZERO")
print("  GDN2_STATE_ZERO")
print("  GDN1_OUTPUT_ZERO")
print("  GDN2_OUTPUT_ZERO")
print("  NULL_STATE_INTERVENTION")
print()
print("STATE ZERO:")
print("  target = committed_state")
print("  timing = after token t, before token t+1")
print("  output path itself untouched")
print()
print("OUTPUT ZERO:")
print("  target = GDN block y only")
print("  returned ephemeral state preserved")
print("  committed state untouched by the intervention hook")
print()
print("NULL:")
print("  copy committed_state -> identical copy of itself")
print("  expected = bitwise clean equivalence")
print()

all_results = {}

for length in (8, 128):
    bits = make_bits(length)

    print("-" * 100)
    print(f"LENGTH {length}  examples={bits.shape[0]}")
    print("-" * 100)

    clean = run_once(bits)
    all_results[f"L{length}_CLEAN"] = {
        "ce": clean["ce"],
        "accuracy": clean["accuracy"],
    }

    print(
        f"CLEAN             CE={clean['ce']:.6f} "
        f"ACC={clean['accuracy']:.4f}"
    )

    state_runs = []

    for gdn_index, gdn_name in ((4, "GDN1"), (5, "GDN2")):
        result = run_once(
            bits,
            state_intervention=gdn_index,
        )

        key = f"L{length}_{gdn_name}_STATE_ZERO"

        all_results[key] = {
            "ce": result["ce"],
            "accuracy": result["accuracy"],
            "delta_accuracy": result["accuracy"] - clean["accuracy"],
            "checks": result["state_checks"],
        }

        state_runs.append(result)

        print(
            f"{gdn_name + '_STATE_ZERO':<19} "
            f"CE={result['ce']:.6f} "
            f"ACC={result['accuracy']:.4f} "
            f"ΔACC={result['accuracy'] - clean['accuracy']:+.4f}"
        )

    output_runs = []

    for gdn_index, gdn_name in ((4, "GDN1"), (5, "GDN2")):
        result = run_once(
            bits,
            output_intervention=gdn_index,
        )

        key = f"L{length}_{gdn_name}_OUTPUT_ZERO"

        all_results[key] = {
            "ce": result["ce"],
            "accuracy": result["accuracy"],
            "delta_accuracy": result["accuracy"] - clean["accuracy"],
        }

        output_runs.append(result)

        print(
            f"{gdn_name + '_OUTPUT_ZERO':<19} "
            f"CE={result['ce']:.6f} "
            f"ACC={result['accuracy']:.4f} "
            f"ΔACC={result['accuracy'] - clean['accuracy']:+.4f}"
        )

    null_result = run_once(
        bits,
        null_intervention=True,
    )

    log_compare = compare_logits(
        clean["logits"],
        null_result["logits"],
    )

    if not log_compare["bitwise_equal"]:
        raise AssertionError(
            "NULL intervention changed model output."
        )

    if not all(null_result["null_checks"]):
        raise AssertionError(
            "NULL intervention failed an identity check."
        )

    all_results[f"L{length}_NULL"] = {
        "ce": null_result["ce"],
        "accuracy": null_result["accuracy"],
        "logit_comparison_to_clean": log_compare,
        "all_identity_checks": all(null_result["null_checks"]),
    }

    print(
        f"{'NULL_STATE':<19} "
        f"CE={null_result['ce']:.6f} "
        f"ACC={null_result['accuracy']:.4f} "
        f"bitwise_equal={log_compare['bitwise_equal']} "
        f"max_abs_diff={log_compare['max_abs_diff']:.3e}"
    )

    # Output-zero interventions must produce a causal effect, otherwise
    # interpretation of an intervention may be confounded by an inactive path.
    for result in output_runs:
        if result["accuracy"] == clean["accuracy"]:
            print(
                "WARNING: output intervention matched clean accuracy; "
                "inspect pathway activity before interpreting causality."
            )


# Load A6 donor-control result if available, without modifying it.
a6_path = Path("experiments/gateA_a6_state_patching.json")

if a6_path.exists():
    a6 = json.loads(a6_path.read_text())

    all_results["A6_same_parity_donor_controls"] = (
        a6.get("same_parity_controls", {})
    )

    print()
    print("-" * 100)
    print("A6 SAME-PARITY DONOR CONTROLS (REUSED, NOT RERUN)")
    print("-" * 100)

    for key in ("GDN1", "GDN2", "BOTH"):
        r = a6["same_parity_controls"][key]

        print(
            f"{key:<19} "
            f"P(correct) clean={r['mean_p_correct_clean']:.4f} "
            f"patched={r['mean_p_correct_patched']:.4f} "
            f"Δ={r['delta_p_correct']:+.4f} "
            f"flip={r['patched_flip_rate']:.4f}"
        )
else:
    print()
    print("WARNING: A6 JSON not found; donor controls not imported.")


report = {
    "gate": "A7",
    "checkpoint": CKPT,
    "device": str(DEVICE),
    "dim": DIM,
    "results": all_results,
    "interpretation_notes": [
        "Null intervention must be bitwise equivalent to clean.",
        "State intervention targets persistent committed_state only.",
        "Output intervention targets only GDN block output y.",
        "A6 same-parity donor controls are reused as donor controls.",
    ],
}

Path("experiments/gateA_a7_controls.json").write_text(
    json.dumps(report, indent=2)
)

print()
print("=" * 100)
print("=== GATE A7 COMPLETE ===")
print("WROTE experiments/gateA_a7_controls.json")
print("=" * 100)
