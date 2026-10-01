
from dataclasses import dataclass, field

import torch


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
        RESTART     : reset live h to h0
        HALT        : terminate execution
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

    def execute(
        self,
        h0,
        memory,
        C_prev,
    ):
        """
        Execute one token.

        This first implementation establishes the execution-state
        machinery. Controller routing is performed per execution.
        """

        # Live state starts from h0.
        h = h0.clone()

        # Explicit execution bookkeeping.
        execution_count = 0
        pointer = 0
        just_restarted = False

        action_history = []

        halted = False
        safety_cap_hit = False

        # Persistent inputs are intentionally kept untouched.
        frozen_memory = memory
        frozen_C_prev = C_prev

        while execution_count < self.max_execution_depth:

            # Controller observes the current live state.
            controller_out = self.controller(h)

            # Current implementation uses the controller outputs
            # directly; legality masking is applied by the caller
            # during the finalized routing implementation.
            advance_logits = controller_out["advance"]
            repeat_logits = controller_out["repeat"]
            restart_logits = controller_out["restart"]
            halt_logits = controller_out["halt"]

            # Deterministic provisional action:
            # select the highest-scoring available action.
            candidates = torch.cat(
                [
                    advance_logits,
                    repeat_logits.unsqueeze(-1),
                    restart_logits.unsqueeze(-1),
                    halt_logits.unsqueeze(-1),
                ],
                dim=-1,
            )

            action = torch.argmax(
                candidates,
                dim=-1,
            )

            # Batch-size-1/per-sequence execution is the current
            # implementation assumption.
            action_id = int(action[0].item())

            num_blocks = len(self.block_bank.blocks)

            if action_id < num_blocks:
                # ADVANCE(j')
                block_index = action_id

                h = self.block_bank(
                    block_index,
                    h.unsqueeze(1),
                ).squeeze(1)

                pointer = block_index
                just_restarted = False

                action_history.append(
                    ("ADVANCE", block_index)
                )

            elif action_id == num_blocks:
                # REPEAT
                h = self.block_bank(
                    pointer,
                    h.unsqueeze(1),
                ).squeeze(1)

                just_restarted = False

                action_history.append(
                    ("REPEAT", pointer)
                )

            elif action_id == num_blocks + 1:
                # RESTART
                h = h0.clone()

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

            # Separate convergence signal.
            # Controller HALT remains independent.
            if self.convergence is not None:
                with torch.no_grad():
                    if self.convergence.converged(h).all().item():
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
