"""Versioned research-only complete-program and graph-preserving receiver adapter.

No training loop, optimizer step, estimator or production semantic implementation.
All operator execution and commits use the current production complete-token path.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.func import functional_call
import torch.nn.functional as F

from src.agt import AGTv3
import src.agt as agt_module
from src.masks import build_execution_mask

ROOT = Path(__file__).resolve().parents[3]
BACKBONE = ROOT / "experiments/runs/stage1_parity_seed3305_best.pt"
BACKBONE_SHA = "53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3"
ATOL, RTOL = 1e-6, 1e-5


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def fields(value, prefix=""):
    """Complete tensor/metadata topology; returns ORIGINAL tensors, never detaches."""
    if torch.is_tensor(value) or value is None or isinstance(value, (str, int, float, bool)):
        return {prefix: value}
    items = value.items() if isinstance(value, dict) else vars(value).items()
    result = {}
    for key, item in items:
        result.update(fields(item, f"{prefix}/{key}"))
    return result


def snapshot(value):
    """Observation only. Never used as a computational handoff."""
    return {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in fields(value).items()}


def assert_equal(left, right, exact=False):
    if left.keys() != right.keys():
        raise AssertionError("Complete state/metadata schema differs")
    for key, a in left.items():
        b = right[key]
        if torch.is_tensor(a):
            assert torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype, key
            assert (torch.equal(a, b) if exact else torch.allclose(a, b, atol=ATOL, rtol=RTOL)), key
        else:
            assert type(a) is type(b) and a == b, (key, a, b)


def assert_disjoint(a, b):
    x, y = fields(a), fields(b)
    assert x.keys() == y.keys()
    for key, value in x.items():
        other = y[key]
        if torch.is_tensor(value) and value.numel():
            assert value.untyped_storage().data_ptr() != other.untyped_storage().data_ptr(), key


def detached_state_fixture(state):
    """NEGATIVE-CONTROL fixture only. Not an admissible study handoff."""
    result = state.clone()
    result.s = result.s.detach()
    result.memory = result.memory.detach()
    for cache in result.ctx.kv.values():
        cache.committed_keys = cache.committed_keys.detach() if cache.committed_keys is not None else None
        cache.committed_values = cache.committed_values.detach() if cache.committed_values is not None else None
    for cache in result.ctx.swa.values():
        cache.committed_keys = cache.committed_keys.detach() if cache.committed_keys is not None else None
        cache.committed_values = cache.committed_values.detach() if cache.committed_values is not None else None
    for cache in result.ctx.gdn.values():
        cache.committed_state = cache.committed_state.detach() if cache.committed_state is not None else None
        cache.committed_conv_buffer = cache.committed_conv_buffer.detach() if cache.committed_conv_buffer is not None else None
    return result


@dataclass(frozen=True)
class ResearchProgram:
    name: str
    actions: tuple[int, ...]

    def validate(self):
        assert type(self.actions) is tuple, "Immutable tuple required"
        pointer, p, restarted = 0, 0, False
        rows = []
        for d, action in enumerate(self.actions):
            if type(action) is not int or not 0 <= action < 10 or p >= 8:
                raise ValueError("Invalid/surplus successor action")
            mask = build_execution_mask(pointer, p, restarted, 8)
            if not bool(mask[action]):
                raise ValueError(f"Illegal action {action} at decision {d}")
            block = action if action < 8 else pointer if action == 8 else None
            before = pointer
            if action < 8:
                pointer, restarted, p = action, False, p + 1
            elif action == 8:
                restarted, p = False, p + 1
            else:
                pointer, restarted = 0, True
            rows.append(dict(decision=d, action=action, block=block,
                             pointer_before=before, pointer_after=pointer,
                             execution_after=p, legal_mask=mask.tolist()))
        if p != 8:
            raise ValueError("Incomplete program; no continuation/fallback permitted")
        return rows


P0 = ResearchProgram("keyed_register_v1/B", (0, 1, 2, 3, 4, 5, 6, 7))
P1 = ResearchProgram("keyed_register_v1/A", (0, 1, 2, 3, 8, 5, 6, 7))
PR = ResearchProgram("keyed_register_v1/R", (0, 1, 2, 4, 8, 5, 6, 7))
PROGRAMS = (P0, P1, PR)


class HistoricalFixtureLedgerUnused:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.rows = []
        self.attempted = self.completed = self.failed = self.blocks = self.commits = 0
        self.block_completed = self.commit_attempts = self.flushed = 0
        self.backward_evaluations = self.warmups = 0
        self.optimizer_updates = 0
        # Carry every earlier disposable fixture attempt; never reset accounting.
        record_path = self.directory / "resource_ledger.json"
        trace_path = self.directory / "fixture_token_traces.jsonl"
        if record_path.exists():
            old = json.loads(record_path.read_text())
            self.attempted = old["attempted_tokens"]
            self.completed = old["completed_tokens"]
            self.failed = old["failed_tokens"]
            self.blocks = old["selected_bank_calls"]
            self.block_completed = old["completed_bank_calls"]
            self.commits = old["terminal_commits"]
            self.commit_attempts = old["attempted_terminal_commits"]
            self.backward_evaluations = old["backward_evaluations"]
            self.rows = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
            self.flushed = len(self.rows)

    def begin(self, case, program, device, token):
        if self.attempted >= 512 or self.blocks + 8 > 4096:
            raise RuntimeError("Prerequisite resource ceiling reached BEFORE next call")
        self.attempted += 1
        row = dict(attempt=self.attempted, case=case, program=program.name,
                   device=str(device), token=token, status="attempted", decisions=[], commits=0)
        self.rows.append(row)
        return row

    def persist(self):
        record = dict(attempted_tokens=self.attempted, completed_tokens=self.completed,
                      failed_tokens=self.failed, selected_bank_calls=self.blocks,
                      completed_bank_calls=self.block_completed, attempted_terminal_commits=self.commit_attempts,
                      terminal_commits=self.commits, backward_evaluations=self.backward_evaluations,
                      optimizer_updates=self.optimizer_updates, warmup_tokens=self.warmups,
                      token_ceiling=512, block_ceiling=4096)
        (self.directory / "resource_ledger.json").write_text(json.dumps(record, indent=2) + "\n")
        with (self.directory / "fixture_token_traces.jsonl").open("a") as f:
            for row in self.rows[self.flushed:]:
                f.write(json.dumps(row) + "\n")
        self.flushed = len(self.rows)


def load_research_model(device):
    if sha(BACKBONE) != BACKBONE_SHA:
        raise RuntimeError("Archived backbone hash mismatch")
    checkpoint = torch.load(BACKBONE, map_location="cpu", weights_only=True)
    weights = {k: v for k, v in checkpoint["state_dict"].items() if not k.startswith("controller.")}
    torch.manual_seed(3305)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=2)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    assert set(missing) == {k for k in model.state_dict() if k.startswith("controller.")} and not unexpected
    model.to(device=device, dtype=torch.float32).eval()
    model.executor.max_execution_depth = 8
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith("controller."))
    return model


def fixture_encoding(bits, device):
    # Existing frozen Task-A encoding; no new study panel or experimental noise.
    g = torch.Generator(device="cpu").manual_seed(3002)
    features = torch.randn(2, 64, generator=g, dtype=torch.float32).to(device)
    b = torch.tensor(bits, dtype=torch.long, device=device)
    return features[b].unsqueeze(0), b.cumsum(0).remainder(2)


class CompleteProgramModule(nn.Module):
    """Registered wrapper permits strict functional receiver parameter substitution."""
    def __init__(self, model, ledger):
        super().__init__()
        self.model = model
        self.ledger = ledger
        self.ablate_fifth = False

    def initial(self, x):
        return self.model.initial_persistent_state(1, x.device, x.dtype)

    def forward(self, x, parent, program, case, reference=False):
        expected = program.validate()
        if reference and program != P0:
            raise ValueError("Canonical reference only accepts research P0")
        row = self.ledger.begin(case, program, x.device, parent.tokens_committed)
        parent_snapshot = snapshot(parent)
        row["incoming_metadata"] = {k: ({"shape": list(v.shape), "dtype": str(v.dtype)} if torch.is_tensor(v) else v) for k, v in fields(parent).items()}
        ex = self.model.executor
        saved = (ex.controller, ex._run_block, ex.candidate_successors, agt_module.terminal_commit)
        handles = []
        decisions, executed = [], []
        current = {"index": 0}

        def provider(**features):
            d = current["index"]
            if d >= len(expected):
                raise AssertionError("Exhausted script; no fallback")
            spec = expected[d]
            assert int(features["pointer_one_hot"].argmax(-1)) == spec["pointer_before"]
            assert bool(features["budget"].eq(0).all())
            mask = torch.tensor(spec["legal_mask"], dtype=torch.bool, device=features["h"].device)
            logits = features["h"].new_full((1, 11), -100.)
            logits[:, spec["action"]] = 100.
            selected = int(logits[:, :10].masked_fill(~mask[:10], -torch.inf).argmax(-1))
            assert selected == spec["action"]
            decisions.append(dict(spec, selected=selected))
            current["index"] += 1
            return {"logits": logits}

        def call_block(block_index, h, h0, memory, context, **kwargs):
            spec = expected[len(executed)]
            assert block_index == spec["block"], "Selected/executed mismatch"
            assert len(decisions) == len(executed) + 1
            before = snapshot(context)
            anchor = h0.detach().clone()
            self.ledger.blocks += 1
            assert self.ledger.blocks <= self.ledger.block_ceiling
            output = saved[1](block_index, h, h0, memory, context, **kwargs)
            assert_equal(before, snapshot(context), exact=True)
            assert torch.equal(anchor, h0)
            suppress = self.ablate_fifth and len(executed) == 4
            executed.append(block_index)
            self.ledger.block_completed += 1
            if suppress:
                row["suppressed_decision"] = 4
                return h
            return output

        def forbidden_candidates(*args, **kwargs):
            raise AssertionError("Exhaustive candidates are forbidden for complete programs")

        def commit(state, **kwargs):
            before = {k: v for k, v in fields(state).items() if k.endswith("/version")}
            self.ledger.commit_attempts += 1
            output, plan = saved[3](state, **kwargs)
            after = {k: v for k, v in fields(output).items() if k.endswith("/version")}
            assert before.keys() == after.keys()
            assert all(after[k] == before[k] + 1 for k in before)
            assert len(plan.entries) == 7 and output.tokens_committed == parent.tokens_committed + 1
            assert not output._consumed and state._consumed
            row["commits"] += 1
            self.ledger.commits += 1
            row["commit"] = dict(token_index=plan.token_index, entry_count=len(plan.entries),
                                 versions_before=before, versions_after=after)
            return output, plan

        try:
            ex.controller, ex._run_block = provider, call_block
            ex.candidate_successors = forbidden_candidates
            agt_module.terminal_commit = commit
            if reference:
                def hook(index):
                    def pre(module, args, kwargs):
                        self.ledger.blocks += 1
                        assert self.ledger.blocks <= self.ledger.block_ceiling
                        before = snapshot(kwargs["C_prev"]) if kwargs["C_prev"] is not None else {}
                        module._fixture_purity_before = before
                    def post(module, args, kwargs, result):
                        assert_equal(module._fixture_purity_before,
                                     snapshot(kwargs["C_prev"]) if kwargs["C_prev"] is not None else {}, exact=True)
                        del module._fixture_purity_before
                        assert index == len(executed)
                        executed.append(index)
                        decisions.append(dict(expected[index], selected=index))
                        self.ledger.block_completed += 1
                    return pre, post
                for i, block in enumerate(self.model.blocks.blocks):
                    pre, post = hook(i)
                    handles.extend([block.register_forward_pre_hook(pre, with_kwargs=True),
                                    block.register_forward_hook(post, with_kwargs=True)])
                out = self.model.step(x, parent, x.new_zeros(1, 64))
            else:
                out = self.model.adaptive_differentiable_step(x, parent, x.new_zeros(1, 64),
                                                              budget=x.new_zeros(1, 2), training=False)
                assert out["execution_count"] == 8
                expected_history = [("ADVANCE", d["action"]) if d["action"] < 8 else
                                    ("REPEAT", d["block"]) for d in expected]
                assert out["action_history"] == expected_history
            assert executed == [d["block"] for d in expected] and len(decisions) == 8
            assert row["commits"] == 1
            assert bool(torch.isfinite(out["logits"]).all()), "Nonfinite token output"
            row["returned_metadata"] = {k: ({"shape": list(v.shape), "dtype": str(v.dtype)} if torch.is_tensor(v) else v) for k, v in fields(out["state"]).items()}
            assert_equal(parent_snapshot, snapshot(parent), exact=True)
            assert_disjoint(parent, out["state"])
            row.update(status="completed", decisions=decisions, executed_blocks=executed, D=8, P=8,
                       restart_count=0, repeat_count=sum(a == 8 for a in program.actions),
                       termination="execution_budget_reached", logits=out["logits"].detach().cpu().tolist())
            self.ledger.completed += 1
            return out
        except BaseException as error:
            self.ledger.failed += 1
            row.update(status="failed", error=f"{type(error).__name__}: {error}", decisions=decisions,
                       executed_blocks=executed)
            raise
        finally:
            ex.controller, ex._run_block, ex.candidate_successors, agt_module.terminal_commit = saved
            for handle in handles:
                handle.remove()
            self.ledger.persist()

