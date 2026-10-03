
import torch
import torch.nn as nn

from .blocks import HeterogeneousBlockBank
from .commit import TerminalCommit
from .controller import Controller
from .convergence import ConvergenceGate
from .executor import AdaptiveExecutor
from .memory import PersistentMemory
from .output_head import OutputHead
from .prelude import Prelude
from .blocks import AGTBlock
from .terminal_commit import (
    AGTPersistentState,
    MemoryWrite,
    block_context,
    terminal_commit,
)


class AGTv3(nn.Module):
    """
    AGT-v3 top-level assembly.

    Macro pipeline:

        H_{t-1}
          -> Prelude
          -> adaptive execution
          -> TerminalCommit
          -> H_t
          -> OutputHead
          -> p(x_{t+1} | H_t)

    Adaptive routing is intentionally not finalized here yet.
    """

    def __init__(
        self,
        dim: int = 256,
        num_blocks: int = 8,
        num_memory_slots: int = 16,
        vocab_size: int = 50257,
    ):
        super().__init__()

        self.dim = dim
        self.num_blocks = num_blocks
        self.num_memory_slots = num_memory_slots
        self.vocab_size = vocab_size

        self.memory = PersistentMemory(
            dim=dim,
            num_slots=num_memory_slots,
        )

        self.prelude = Prelude(
            dim=dim,
        )

        # The terminal-commit contract includes a dedicated Prelude SWA
        # context entry. This block owns that context; Prelude itself remains
        # the macro input constructor.
        self.prelude_block = AGTBlock(
            dim=dim,
            species="SWA",
        )

        self.blocks = HeterogeneousBlockBank(
            dim=dim,
        )

        self.swa_window = self.blocks.blocks[0].core.window_size

        self.controller = Controller(
            dim=dim,
            num_blocks=num_blocks,
        )

        self.convergence = ConvergenceGate(
            dim=dim,
        )

        self.executor = AdaptiveExecutor(
            block_bank=self.blocks,
            controller=self.controller,
            convergence=self.convergence,
            max_execution_depth=256,
        )

        self.commit = TerminalCommit(
            dim=dim,
        )

        # Blueprint memory write lives at terminal commit only.
        self.memory_write = MemoryWrite(
            dim=dim,
            mem_dim=256,
        )

        self.output_head = OutputHead(
            dim=dim,
            vocab_size=vocab_size,
        )

    def initial_state(
        self,
        batch_size,
        device=None,
        dtype=None,
    ):
        """
        Create the persistent state required before processing
        the first token.
        """

        H_prev = torch.zeros(
            batch_size,
            self.dim,
            device=device,
            dtype=dtype,
        )

        M = self.memory.initial_state(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        C_prev = torch.zeros(
            batch_size,
            self.dim,
            device=device,
            dtype=dtype,
        )

        return H_prev, M, C_prev

    def initial_persistent_state(
        self,
        batch_size,
        device=None,
        dtype=None,
    ):
        """
        Create the locked persistent token state:

            H_0 = (s_0, M_1, C_0)

        where C_0 is the species-typed ExecutionCache.
        """
        s0 = torch.zeros(
            self.dim,
            device=device,
            dtype=dtype,
        )

        M1 = self.memory.initial_state(
            batch_size=1,
            device=device,
            dtype=dtype,
        )[0]

        return AGTPersistentState.initial(
            s0=s0,
            M1=M1,
            species_order=self.blocks.species_order,
            swa_window=self.swa_window,
            batch=batch_size,
            with_prelude=True,
        )

    def step(
        self,
        x_t,
        state,
        C_prev,
    ):
        """
        Sequential fixed-route token step.

        This milestone deliberately does NOT invoke AdaptiveExecutor.
        Every block in the locked bank is visited once in deterministic
        bank order. The caller receives a new persistent state.

        C_prev remains an explicit compatibility input because the current
        Prelude API still defines it as a [B, D] tensor; no new C_t update
        rule is invented here.
        """
        if x_t.ndim != 2 or x_t.shape[-1] != self.dim:
            raise ValueError(
                f"x_t must be [B, {self.dim}]."
            )

        if state.s.shape != x_t.shape:
            raise ValueError(
                "state.s shape must match x_t."
            )

        if state.memory.shape[0] != x_t.shape[0]:
            raise ValueError(
                "state.memory batch size must match x_t."
            )

        if C_prev.ndim != 2 or C_prev.shape != x_t.shape:
            raise ValueError(
                "C_prev must have shape [B, dim]."
            )

        if state.tokens_committed < 0:
            raise ValueError(
                "tokens_committed must be non-negative."
            )

        # --------------------------------------------------------
        # Snapshot first. No caller-owned persistent cache may be
        # mutated by in-flight execution or by terminal commit.
        # --------------------------------------------------------
        work_state = state.clone()

        # --------------------------------------------------------
        # Persistent memory read for this token.
        # --------------------------------------------------------
        memory_read, memory_attention = self.memory.read(
            work_state.s,
            work_state.memory,
        )

        # --------------------------------------------------------
        # Prelude -> h0.
        # --------------------------------------------------------
        h0 = self.prelude(
            x_t=x_t,
            H_prev=work_state.s,
            memory_read=memory_read,
            C_prev=C_prev,
        )

        # Block API is token-shaped: [B, 1, D].
        h = h0.unsqueeze(1)
        h0_3 = h0.unsqueeze(1)
        R_c = memory_read.unsqueeze(1)

        # --------------------------------------------------------
        # Fixed deterministic route:
        # [SWA, SWA, GQA, GQA, GDN, GDN, FFN, FFN]
        #
        # Context supplied here is committed context only.
        # Block internals construct their ephemeral view.
        # --------------------------------------------------------
        for i, blk in enumerate(self.blocks.blocks):
            cache = block_context(
                work_state.ctx,
                i,
                blk.species,
            )

            if blk.species == "FFN":
                out = blk(
                    h,
                    h0=h0_3,
                    R=R_c,
                    C_prev=None,
                )
            else:
                out = blk(
                    h,
                    h0=h0_3,
                    R=R_c,
                    C_prev=cache,
                )

            if blk.species == "GDN":
                h, _ephemeral_state = out
            else:
                h = out

        h_terminal = h

        # --------------------------------------------------------
        # Exactly one terminal commit.
        # Prelude commit entry is based on the token input x_t,
        # matching the existing terminal_commit contract.
        # --------------------------------------------------------
        new_state, plan = terminal_commit(
            work_state,
            bank=self.blocks,
            prelude_block=self.prelude_block,
            prelude_input=x_t.unsqueeze(1),
            final_norm=self.commit,
            memory_write=self.memory_write,
            h_terminal=h_terminal,
            h0=h0_3,
            R_c=R_c,
            memory_keys=self.memory.memory_keys(work_state.memory),
            token_index=work_state.tokens_committed,
        )

        logits = self.output_head(new_state.s)

        return {
            "state": new_state,
            "h0": h0,
            "h_terminal": h_terminal.squeeze(1),
            "H_t": new_state.s,
            "logits": logits,
            "memory_read": memory_read,
            "memory_attention": memory_attention,
            "commit_plan": plan,
        }


    def adaptive_differentiable_step(
        self,
        x_t,
        state,
        C_prev,
        memory_mean=None,
        budget=None,
        tau=1.0,
        training=True,
    ):
        """
        One-token differentiable AGT-v3 Stage-3 path.

        Macro path:

            persistent state
                -> memory read
                -> Prelude
                -> differentiable executor
                -> terminal commit
                -> output head

        This is intentionally separate from:
            - step(): frozen fixed-route reference
            - adaptive_step(): legacy discrete/argmax adaptive path

        Stage-3 scope:
            - fixed execution depth supplied by executor
            - ST-Gumbel routing
            - HALT excluded from successor mixing
            - terminal commit remains atomic
            - no learned halting signal
        """
        if x_t.ndim != 2 or x_t.shape[-1] != self.dim:
            raise ValueError(f"x_t must be [B, {self.dim}].")

        if state.s.shape != x_t.shape:
            raise ValueError("state.s shape must match x_t.")

        if state.memory.shape[0] != x_t.shape[0]:
            raise ValueError("state.memory batch size must match x_t.")

        if C_prev.ndim != 2 or C_prev.shape != x_t.shape:
            raise ValueError("C_prev must have shape [B, dim].")

        work_state = state.clone()

        memory_read, memory_attention = self.memory.read(
            work_state.s,
            work_state.memory,
        )

        h0 = self.prelude(
            x_t=x_t,
            H_prev=work_state.s,
            memory_read=memory_read,
            C_prev=C_prev,
        )

        # Source-aligned controller slot summary:
        # mean over the persistent memory slots in d_M=256 space.
        if memory_mean is None:
            memory_mean = work_state.memory.mean(dim=1)
        else:
            if memory_mean.ndim != 2 or memory_mean.shape != (
                x_t.shape[0],
                256,
            ):
                raise ValueError(
                    "memory_mean must have shape [B, 256]."
                )

        if budget is None:
            budget = torch.zeros(
                x_t.shape[0],
                2,
                device=x_t.device,
                dtype=x_t.dtype,
            )
        else:
            if budget.ndim != 2 or budget.shape != (x_t.shape[0], 2):
                raise ValueError(
                    "budget must have shape [B, 2]."
                )

        execution = self.executor.execute_differentiable(
            h0=h0,
            memory=memory_read,
            C_prev=C_prev,
            context=work_state.ctx,
            memory_mean=memory_mean,
            budget=budget,
            tau=tau,
            training=training,
        )

        h_terminal = execution.h_terminal.unsqueeze(1)
        h0_3 = h0.unsqueeze(1)
        R_c = memory_read.unsqueeze(1)

        new_state, plan = terminal_commit(
            work_state,
            bank=self.blocks,
            prelude_block=self.prelude_block,
            prelude_input=x_t.unsqueeze(1),
            final_norm=self.commit,
            memory_write=self.memory_write,
            h_terminal=h_terminal,
            h0=h0_3,
            R_c=R_c,
            memory_keys=self.memory.memory_keys(work_state.memory),
            token_index=work_state.tokens_committed,
        )

        logits = self.output_head(new_state.s)

        return {
            "state": new_state,
            "h0": h0,
            "h_terminal": execution.h_terminal,
            "H_t": new_state.s,
            "logits": logits,
            "memory_read": memory_read,
            "memory_attention": memory_attention,
            "commit_plan": plan,
            "execution_count": execution.execution_count,
            "action_history": execution.action_history,
            "halted": execution.halted,
            "safety_cap_hit": execution.safety_cap_hit,
        }

    def adaptive_step(
        self,
        x_t,
        state,
        C_prev,
    ):
        """
        One-token adaptive execution path using the persistent-state contract.

        This remains separate from step(), which is the frozen fixed-route
        reference path. The AdaptiveExecutor supplies h_terminal; the same
        locked terminal commit then produces the new persistent state and logits.
        """
        if x_t.ndim != 2 or x_t.shape[-1] != self.dim:
            raise ValueError(f"x_t must be [B, {self.dim}].")

        if state.s.shape != x_t.shape:
            raise ValueError("state.s shape must match x_t.")

        if state.memory.shape[0] != x_t.shape[0]:
            raise ValueError("state.memory batch size must match x_t.")

        if C_prev.ndim != 2 or C_prev.shape != x_t.shape:
            raise ValueError("C_prev must have shape [B, dim].")

        work_state = state.clone()

        memory_read, memory_attention = self.memory.read(
            work_state.s,
            work_state.memory,
        )

        h0 = self.prelude(
            x_t=x_t,
            H_prev=work_state.s,
            memory_read=memory_read,
            C_prev=C_prev,
        )

        execution = self.executor.execute(
            h0=h0,
            memory=memory_read,
            C_prev=C_prev,
            context=work_state.ctx,
        )

        h_terminal = execution.h_terminal.unsqueeze(1)
        h0_3 = h0.unsqueeze(1)
        R_c = memory_read.unsqueeze(1)

        new_state, plan = terminal_commit(
            work_state,
            bank=self.blocks,
            prelude_block=self.prelude_block,
            prelude_input=x_t.unsqueeze(1),
            final_norm=self.commit,
            memory_write=self.memory_write,
            h_terminal=h_terminal,
            h0=h0_3,
            R_c=R_c,
            memory_keys=self.memory.memory_keys(work_state.memory),
            token_index=work_state.tokens_committed,
        )

        logits = self.output_head(new_state.s)

        return {
            "state": new_state,
            "h0": h0,
            "h_terminal": execution.h_terminal,
            "H_t": new_state.s,
            "logits": logits,
            "memory_read": memory_read,
            "memory_attention": memory_attention,
            "commit_plan": plan,
            "execution_count": execution.execution_count,
            "action_history": execution.action_history,
            "halted": execution.halted,
            "safety_cap_hit": execution.safety_cap_hit,
        }

    def forward(
        self,
        x_t,
        H_prev,
        memory_read,
        C_prev,
    ):
        """
        Assemble one token's basic macro path.

        Full adaptive execution is connected later.
        """

        h0 = self.prelude(
            x_t=x_t,
            H_prev=H_prev,
            memory_read=memory_read,
            C_prev=C_prev,
        )

        execution = self.executor.execute(
            h0=h0,
            memory=memory_read,
            C_prev=C_prev,
        )

        h_terminal = execution.h_terminal

        H_t = self.commit(
            h_terminal
        )

        logits = self.output_head(
            H_t
        )

        return {
            "h0": h0,
            "h_terminal": h_terminal,
            "H_t": H_t,
            "logits": logits,
            "execution_count": execution.execution_count,
            "action_history": execution.action_history,
            "halted": execution.halted,
            "safety_cap_hit": execution.safety_cap_hit,
        }
