"""Disposable prerequisites only. No study data, optimizer steps or parameter updates."""
import copy
import json
from pathlib import Path
import time

import pytest
import torch

from receiver_anchored_v1 import (ROOT, P0, P1, ResearchProgram, ResourceLedger,
    CompleteProgramModule, DetachedReceiver, load_research_model, fixture_encoding,
    fields, snapshot, assert_equal, assert_disjoint, detached_state_fixture,
    rollout, episode, native_loss, arm_loss)
from statistical_rules_v2 import make_bank, stratified_means, quantile, ablation_rule, latency_rule

OUT = Path(__file__).resolve().parent
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FACTS = {}


def fact(name, value):
    FACTS[name] = value
    (OUT / "test_measurements.json").write_text(json.dumps(FACTS, indent=2) + "\n")


def gradient_rows(parameters, gradients):
    return {name: dict(connected=g is not None,
                      l2=float(g.norm()) if g is not None else None,
                      max_abs=float(g.abs().max()) if g is not None else None,
                      finite=bool(torch.isfinite(g).all()) if g is not None else None,
                      exact_zero=bool(g.eq(0).all()) if g is not None else None)
            for (name, _), g in zip(parameters, gradients)}


def total_norm(gradients):
    return float(torch.sqrt(sum(g.square().sum() for g in gradients if g is not None)))


@pytest.fixture(scope="session")
def ledger():
    result = ResourceLedger(OUT)
    yield result
    result.persist()


@pytest.fixture(scope="session", params=DEVICES)
def wrapper(request, ledger):
    model = load_research_model(request.param)
    before = snapshot(dict(model.named_parameters()))
    buffers = snapshot(dict(model.named_buffers()))
    result = CompleteProgramModule(model, ledger)
    yield result
    assert_equal(before, snapshot(dict(model.named_parameters())), exact=True)
    assert_equal(buffers, snapshot(dict(model.named_buffers())), exact=True)
    assert all(p.grad is None for p in model.parameters())
    fact("identity/" + request.param, dict(parameters_unchanged=True, buffers_unchanged=True,
                                         parameter_grad_buffers_untouched=True))


def test_descriptor_contract_fail_closed():
    assert [r["block"] for r in P0.validate()] == list(range(8))
    assert [r["block"] for r in P1.validate()] == [0, 1, 2, 3, 3, 5, 6, 7]
    assert sum(r["action"] == 8 for r in P1.validate()) == 1
    for actions in ((8,) + tuple(range(7)), tuple(range(7)), tuple(range(8)) + (8,),
                    (0, 1, 0, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 10)):
        with pytest.raises(ValueError):
            ResearchProgram("fixture/invalid", actions).validate()


def test_p0_archived_control_complete_forward_equivalence(wrapper):
    control = json.loads((ROOT / "experiments/results/gatec_execution_accounting_v1/gatec_canonical_replay.json").read_text())
    x, labels = fixture_encoding(control["bits"], next(wrapper.parameters()).device)
    a, b = wrapper.initial(x), wrapper.initial(x)
    differences = []
    with torch.no_grad():
        for j in range(8):
            ref = wrapper(x[:, j], a, P0, "archival-control/reference", reference=True)
            result = wrapper(x[:, j], b, P0, "archival-control/script")
            assert_equal(snapshot(ref["state"]), snapshot(result["state"]))
            for key in ("h0", "h_terminal", "memory_read", "memory_attention", "logits", "H_t"):
                assert torch.allclose(ref[key], result[key], atol=1e-6, rtol=1e-5), key
            assert ref["commit_plan"].token_index == result["commit_plan"].token_index
            assert len(ref["commit_plan"].entries) == len(result["commit_plan"].entries) == 7
            differences.append(float((ref["logits"] - result["logits"]).abs().max()))
            a, b = ref["state"], result["state"]
        ce = float(torch.nn.functional.cross_entropy(result["logits"], labels[-1:]))
    assert abs(ce - .0014546061865985394) <= 1e-6
    schema = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) if torch.is_tensor(v) else v
              for k, v in fields(b).items()}
    fact("p0_equivalence/" + str(x.device), dict(final_ce=ce, expected_ce=.0014546061865985394,
         per_token_max_logit_difference=differences, complete_state_schema=schema,
         comparison="fresh same-checkpoint full-state reference vs adapter, all 8 tokens; archived CE control",
         historical_full_tensor_equality_claimed=False))


def test_p1_actual_selected_route_and_no_candidates(wrapper):
    x, labels = fixture_encoding([1], next(wrapper.parameters()).device)
    parent = wrapper.initial(x)
    result = wrapper(x[:, 0], parent, P1, "route/P1")
    row = wrapper.ledger.rows[-1]
    assert row["executed_blocks"] == [0, 1, 2, 3, 3, 5, 6, 7]
    assert row["P"] == row["D"] == 8 and row["commits"] == 1
    assert result["state"].ctx.gdn[4].version == 1
    assert result["state"].ctx.gdn[4].committed_state is not None
    fact("p1_route/" + str(x.device), dict(blocks=row["executed_blocks"], D=8, P=8,
          GDN4_still_committed=True, exhaustive_candidate_calls=0))


def test_complete_clone_ownership_and_graph_paths(wrapper):
    x, labels = fixture_encoding([0, 1, 1, 0], next(wrapper.parameters()).device)
    _, parent, _ = rollout(wrapper, x[:, :2], labels[:2], P0, "clone/writer")
    original = snapshot(parent)
    a, b = parent.clone(), parent.clone()
    assert_disjoint(parent, a)
    assert_disjoint(parent, b)
    assert_disjoint(a, b)
    assert_equal(original, snapshot(a), exact=True)
    receiver = DetachedReceiver(wrapper)
    loss, returned, _ = rollout(wrapper, x[:, 2:], labels[2:], P1, "clone/receiver",
                                initial=a, receiver=receiver)
    incoming = [(k, v) for k, v in fields(a).items() if torch.is_tensor(v) and v.requires_grad]
    gradients = torch.autograd.grad(loss, [v for _, v in incoming], allow_unused=True)
    wrapper.ledger.backward_evaluations += 1
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
    assert any(g is not None and bool(g.ne(0).any()) for g in gradients)
    assert_equal(original, snapshot(parent), exact=True)
    assert_equal(original, snapshot(b), exact=True)
    assert returned.tokens_committed == 4
    fact("clone_graph/" + str(x.device), gradient_rows(incoming, gradients))


def test_directed_writer_gradients_native_loss_and_detached_state_control(wrapper):
    x, labels = fixture_encoding([int(c) for c in "0110100110010110"], next(wrapper.parameters()).device)
    n0, _, h0 = rollout(wrapper, x, labels, P0, "gradient/nativeP0", handoff=8)
    n1, _, h1 = rollout(wrapper, x, labels, P1, "gradient/nativeP1", handoff=8)
    receiver = DetachedReceiver(wrapper)
    i0, i1 = h0.clone(), h1.clone()
    c0, _, _ = rollout(wrapper, x[:, 8:], labels[8:], P1, "gradient/0to1", initial=i0, receiver=receiver)
    c1, _, _ = rollout(wrapper, x[:, 8:], labels[8:], P0, "gradient/1to0", initial=i1, receiver=receiver)
    continuation = (c0 + c1) / 2
    named = [(n, p) for n, p in wrapper.model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    gn = torch.autograd.grad((n0 + n1) / 2, params, retain_graph=True, allow_unused=True)
    incoming = [(f"arm{j}{k}", v) for j, state in enumerate((i0, i1))
                for k, v in fields(state).items() if torch.is_tensor(v) and v.requires_grad]
    gs = torch.autograd.grad(continuation, [v for _, v in incoming], retain_graph=True, allow_unused=True)
    direct = torch.autograd.grad(continuation, params, retain_graph=True, allow_unused=True)
    connected = [(v, g) for (_, v), g in zip(incoming, gs) if g is not None]
    via_state = torch.autograd.grad([v for v, _ in connected], params,
                                   grad_outputs=[g for _, g in connected], allow_unused=True)
    wrapper.ledger.backward_evaluations += 4
    for full, chained in zip(direct, via_state):
        assert (full is None) == (chained is None)
        if full is not None:
            assert torch.allclose(full, chained, atol=1e-6, rtol=1e-5)
            assert torch.isfinite(full).all()
    assert total_norm(direct) > 0 and total_norm(gn) > 0
    assert all(not p.requires_grad and p.grad is None for p in receiver.parameters.values())
    assert all(p.grad is None for p in wrapper.model.controller.parameters())
    negative, _, _ = rollout(wrapper, x[:, 8:], labels[8:], P1, "gradient/detached-state-negative",
                             initial=detached_state_fixture(h0), receiver=receiver)
    assert not negative.requires_grad  # Disconnection caused ONLY by the explicit negative-control handoff.
    fact("directed_gradient/" + str(x.device), dict(native_loss=float(((n0+n1)/2).detach()),
        cross_continuation_loss=float(continuation.detach()), native_parameter_l2=total_norm(gn),
        writer_parameter_l2=total_norm(direct), incoming_state_gradients=gradient_rows(incoming, gs),
        native_parameter_gradients=gradient_rows(named, gn), writer_parameter_gradients=gradient_rows(named, direct),
        full_gradient_equals_complete_state_chain_rule=True, receiver_parameter_requires_grad=False,
        receiver_direct_parameter_credit=False, detached_state_negative_loss_requires_grad=negative.requires_grad,
        scope="disposable fixed fixture; no learning/performance claim"))


def test_independent_model_buffer_and_optimizer_moment_ownership(wrapper):
    if str(next(wrapper.parameters()).device) != "cpu":
        pytest.skip("Ownership fixture uses CPU synthetic optimizer moments; GPU gradient coverage is separate")
    common = wrapper.model
    arms = [copy.deepcopy(common), copy.deepcopy(common)]
    optimizers = [torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                  lr=1e-4, weight_decay=0., betas=(.9, .999), eps=1e-8) for model in [common] + arms]
    # Construct synthetic moment fixtures WITHOUT calling optimizer.step or changing parameters.
    for group in optimizers[0].param_groups:
        for p in group["params"]:
            optimizers[0].state[p] = dict(step=torch.tensor(512.), exp_avg=torch.full_like(p, .001),
                                        exp_avg_sq=torch.full_like(p, .002))
    state_dict = copy.deepcopy(optimizers[0].state_dict())
    for opt in optimizers[1:]:
        opt.load_state_dict(copy.deepcopy(state_dict))
    for arm in arms:
        assert arm.executor.block_bank is arm.blocks and arm.executor.controller is arm.controller
        assert_equal(snapshot(dict(common.named_parameters())), snapshot(dict(arm.named_parameters())), exact=True)
        assert_equal(snapshot(dict(common.named_buffers())), snapshot(dict(arm.named_buffers())), exact=True)
        for (n, a), (m, b) in zip(common.named_parameters(), arm.named_parameters()):
            assert n == m and a.data_ptr() != b.data_ptr()
        for (n, a), (m, b) in zip(common.named_buffers(), arm.named_buffers()):
            assert n == m and a.data_ptr() != b.data_ptr()
    triples = zip(*[[p for group in opt.param_groups for p in group["params"]] for opt in optimizers])
    checked = 0
    for triple in triples:
        for key in ("step", "exp_avg", "exp_avg_sq"):
            tensors = [opt.state[p][key] for opt, p in zip(optimizers, triple)]
            assert torch.equal(tensors[0], tensors[1]) and torch.equal(tensors[1], tensors[2])
            assert len({v.data_ptr() for v in tensors}) == 3
            checked += 1
    fact("ownership", dict(independent_models=True, buffers_include_nonpersistent_RoPE=True,
          independent_synthetic_optimizer_moments=True, moment_fields_checked=checked,
          optimizer_steps=0, synthetic_step_metadata=512, historical_optimizer_loaded=False))


def test_objective_arithmetic_and_effective_batch_equal_episode_weighting(ledger):
    a, b = torch.tensor([1., 3.]), torch.tensor([2., 6.])
    ln = native_loss(a, b)
    assert float(ln) == 3.
    assert float(arm_loss(ln, torch.tensor(4.), torch.tensor(8.))) == 4.5
    # Each episode contributes 1/2 regardless of length; clip only after accumulation.
    scalar = torch.tensor(2., requires_grad=True)
    l16, l24 = scalar * 3, scalar * 7
    assert float(torch.autograd.grad((l16+l24)/2, scalar)[0]) == 5.
    ledger.backward_evaluations += 1
    fact("objective_arithmetic", dict(native_example=3., arm_example=4.5,
          effective_batch="mean of two per-episode objectives; one length16 and one length24"))


def test_locked_ablation_statistics_use_four_contrasts_and_linear_quantiles():
    strata = torch.arange(4).repeat_interleave(128)
    bank = make_bank(strata)
    assert bank.replicates == 20000 and bank.seed == 6010
    u = torch.ones((512, 4), dtype=torch.int64)
    a = torch.ones_like(u)
    for group in bank.groups:
        a[group[:16], :] = 0
    result = ablation_rule(u, a, bank)
    assert bool((result["point"] == .125).all()) and result["passes"].all()
    low = torch.ones_like(u)
    for group in bank.groups:
        low[group[:4], :] = 0
    assert not ablation_rule(u, low, bank)["passes"].any()
    assert not ablation_rule(u, u, bank)["passes"].any()
    assert float(quantile(torch.arange(4, dtype=torch.float64), .0125)) == pytest.approx(.0375)
    # The same sealed case indices pair all four contrasts, both suffixes collapsed per case.
    assert make_bank(strata).identity == bank.identity
    fact("synthetic_ablation", dict(bank_identity=bank.identity, point=result["point"].tolist(),
          adjusted_lower=result["lower"].tolist(), passes=result["passes"].tolist(),
          family_size=4, quantile=.0125, source="synthetic robust-case arrays only"))


def test_locked_latency_statistics_and_unassessable_measurements():
    lengths = torch.tensor([16, 24]).repeat_interleave(16)
    parities = torch.tensor([0, 1]).repeat_interleave(8).repeat(2)
    bank = make_bank(lengths*2 + parities)  # Separate seed-6010 profiling bank.
    timed = torch.ones((32, 2, 2, 5), dtype=torch.float64)
    timed[:, 0] *= 1.05
    result = latency_rule(timed, lengths, parities, bank)
    assert torch.allclose(result["point"], torch.full_like(result["point"], 1.05)) and result["passes"].all()
    failed = timed.clone(); failed[:, 0] = 1.11
    assert not latency_rule(failed, lengths, parities, bank)["passes"].any()
    for invalid in (timed[..., :4], torch.full_like(timed, float("nan")), torch.zeros_like(timed)):
        with pytest.raises(ValueError):
            latency_rule(invalid, lengths, parities, bank)
    # Ratio of arithmetic means, deliberately different from mean of ratios.
    varying = torch.ones_like(timed)
    varying[::2, 0] = 1.; varying[1::2, 0] = 9.
    varying[::2, 1] = 1.; varying[1::2, 1] = 3.
    assert torch.allclose(latency_rule(varying, lengths, parities, bank)["point"], torch.full((4,), 2.5, dtype=torch.float64))
    fact("synthetic_latency", dict(bank_identity=bank.identity, ratio=result["point"].tolist(),
          adjusted_upper=result["upper"].tolist(), passes=result["passes"].tolist(),
          family_size=4, quantile=.9875, repeats_averaged_before_bootstrap=5,
          warmup_included=False, source="synthetic arrays; no study profiling"))


@pytest.mark.parametrize("arm", ["T", "C"])
def test_disposable_full_effective_batch_resource_probe(wrapper, arm):
    device = next(wrapper.parameters()).device
    if device.type != "cuda":
        pytest.skip("CUDA-only representative resource probe; CPU tests do not substitute for accelerator timing")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_tokens = wrapper.ledger.attempted
    start = time.perf_counter()
    losses = []
    for length in (16, 24):
        bits = [((j*j + 3*j + 1) % 7) % 2 for j in range(length)]
        x, labels = fixture_encoding(bits, device)
        loss, receiver = episode(wrapper, x, labels, arm, "runtime-fixture/" + arm + f"/L{length}")
        losses.append(loss)
    objective = sum(losses) / 2
    named = [(n, p) for n, p in wrapper.model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(objective, [p for _, p in named], allow_unused=True)
    wrapper.ledger.backward_evaluations += 1
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    assert total_norm(grads) > 0
    # Global clipping is tested on disposable gradient VALUES, never assigned to parameters.
    norm = total_norm(grads)
    factor = min(1., 1. / (norm + 1e-6))
    clipped_norm = total_norm([g * factor if g is not None else None for g in grads])
    assert clipped_norm <= 1.000001
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = wrapper.ledger.attempted - start_tokens
    assert tokens == 112
    fact("resource_probe/" + arm, dict(device=str(device), tokens=tokens,
          forward_and_backward_seconds=elapsed, tokens_per_second=tokens/elapsed,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          peak_reserved_bytes=torch.cuda.max_memory_reserved(), raw_gradient_l2=norm,
          clipped_disposable_gradient_l2=clipped_norm, objective=float(objective.detach()),
          includes_fixture_observation_and_disk_ledger=True, optimizer_steps=0,
          warning="representative disposable batch, not measured full-study or deployment latency"))
