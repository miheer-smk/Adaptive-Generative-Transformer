from dataclasses import dataclass, field

import torch

from .masks import build_execution_mask, apply_action_mask
from .terminal_commit import block_context


@dataclass
class ExecutionResult:
    h_terminal: torch.Tensor
    execution_count: int
    action_history: list = field(default_factory=list)
    halted: bool = False
    safety_cap_hit: bool = False


class AdaptiveExecutor:
    """
    AGT-v3 adaptive within-token executor.

    Persistent state is supplied by the caller and is never mutated
    in-place during execution.

    Actions:

        ADVANCE(j') : execute selected block and move pointer
        REPEAT      : execute current block again
        RESTART     : preserve live h and reset routing position
        HALT        : terminate execution

    `context` is optional for backward compatibility.

    When `context` is supplied, selected blocks receive the same
    AGTBlock interface used by the validated fixed-route path:

        h, h0, R, C_prev

    where:
        h0 = read-only anchor
        R  = memory read
        C_prev = block-specific committed context
    """

    def __init__(
        self,
        block_bank,
        controller,
        convergence=None,
        max_execution_depth=256,
    ):
        self.block_bank = block_bank
        self.controller = controller
        self.convergence = convergence
        self.max_execution_depth = max_execution_depth

    def _run_block(
        self,
        block_index,
        h,
        h0,
        memory,
        context,
    ):
        """
        Execute one bank block using the AGTBlock executor-facing API.

        h        : [B, D]
        h0       : [B, D]
        memory   : [B, D] memory read R
        context  : persistent ExecutionCache or None

        Returns:
            h_next : [B, D]
        """
        block = self.block_bank.blocks[block_index]

        if context is None:
            out = self.block_bank(
                block_index,
                h.unsqueeze(1),
            )
        else:
            cache = block_context(
                context,
                block_index,
                block.species,
            )

            out = self.block_bank(
                block_index,
                h.unsqueeze(1),
                h0=h0.unsqueeze(1),
                R=memory.unsqueeze(1),
                C_prev=cache,
            )

        # GDN returns:
        #     (output, ephemeral_state)
        # All other species return a tensor.
        #
        # Use the output structure rather than assuming every bank
        # implementation exposes a `.species` attribute. Stage-0 tests
        # intentionally use lightweight toy bank objects.
        if isinstance(out, (tuple, list)):
            out = out[0]

        return out.squeeze(1)

    def execute(
        self,
        h0,
        memory,
        C_prev,
        context=None,
    ):
        """
        Execute one token.

        `memory` is the D-dimensional memory read R used by AGTBlock.
        `C_prev` remains the legacy compatibility input.

        `context`, when supplied, is the species-typed committed
        ExecutionCache carried by AGTPersistentState.

        This milestone validates controller-driven single-pass block
        execution. Full repeated-block ephemeral-state semantics remain
        a separate routing milestone.
        """

        h = h0.clone()

        execution_count = 0
        pointer = 0
        just_restarted = False

        action_history = []

        halted = False
        safety_cap_hit = False

        frozen_memory = memory
        frozen_C_prev = C_prev

        del frozen_C_prev  # retained only as an explicit compatibility input

        while execution_count < self.max_execution_depth:

            controller_out = self.controller(h)

            advance_logits = controller_out["advance"]
            repeat_logits = controller_out["repeat"]
            restart_logits = controller_out["restart"]
            halt_logits = controller_out["halt"]

            candidates = torch.cat(
                [
                    advance_logits,
                    repeat_logits.unsqueeze(-1),
                    restart_logits.unsqueeze(-1),
                    halt_logits.unsqueeze(-1),
                ],
                dim=-1,
            )

            mask = build_execution_mask(
                pointer=pointer,
                execution_count=execution_count,
                just_restarted=just_restarted,
                num_blocks=len(self.block_bank.blocks),
            ).to(device=candidates.device)

            masked_candidates = apply_action_mask(
                candidates,
                mask.unsqueeze(0),
            )

            action = torch.argmax(
                masked_candidates,
                dim=-1,
            )

            # Current executor contract assumes batch-size 1.
            action_id = int(action[0].item())

            num_blocks = len(self.block_bank.blocks)

            # Keep the pre-action state for the deterministic convergence gate.
            h_prev = h

            if action_id < num_blocks:
                # ADVANCE(j')
                block_index = action_id

                h = self._run_block(
                    block_index=block_index,
                    h=h,
                    h0=h0,
                    memory=frozen_memory,
                    context=context,
                )

                pointer = block_index
                just_restarted = False

                action_history.append(
                    ("ADVANCE", block_index)
                )

            elif action_id == num_blocks:
                # REPEAT
                h = self._run_block(
                    block_index=pointer,
                    h=h,
                    h0=h0,
                    memory=frozen_memory,
                    context=context,
                )

                just_restarted = False

                action_history.append(
                    ("REPEAT", pointer)
                )

            elif action_id == num_blocks + 1:
                # RESTART intentionally preserves the live hidden state.
                pointer = 0
                just_restarted = True

                action_history.append(
                    ("RESTART", None)
                )

            else:
                # HALT
                halted = True

                action_history.append(
                    ("HALT", None)
                )

                break

            execution_count += 1

            # Deterministic convergence gate uses both previous and current h.
            if self.convergence is not None:
                with torch.no_grad():
                    if self.convergence.converged(h_prev, h).all().item():
                        halted = True

                        action_history.append(
                            ("CONVERGED", None)
                        )

                        break

        if not halted and execution_count >= self.max_execution_depth:
            safety_cap_hit = True

        return ExecutionResult(
            h_terminal=h,
            execution_count=execution_count,
            action_history=action_history,
            halted=halted,
            safety_cap_hit=safety_cap_hit,
        )
