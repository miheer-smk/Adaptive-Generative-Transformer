from dataclasses import dataclass, field
import inspect

import torch

from .masks import build_execution_mask, apply_action_mask
from .terminal_commit import block_context
from .routing import st_gumbel_successor_mix


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
        return_ephemeral=False,
    ):
        """
        Execute one bank block using the AGTBlock executor-facing API.

        h        : [B, D]
        h0       : [B, D]
        memory   : [B, D] memory read R
        context  : persistent ExecutionCache or None

        Args:
            return_ephemeral:
                When False, preserve the original executor contract and
                return only h_next.

                When True, return:
                    (h_next, ephemeral_state)

                For GDN, ephemeral_state is the value produced by the
                block's pure in-flight execution. For all other species,
                ephemeral_state is None.

        Persistent caches are never mutated by this method.
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
                return_ephemeral=return_ephemeral,
            )

        # GDN returns:
        #     (output, ephemeral_state)
        # All other species return a tensor.
        #
        # Preserve the ephemeral state when explicitly requested.
        ephemeral_state = None

        if isinstance(out, (tuple, list)):
            if len(out) != 2:
                raise RuntimeError(
                    "Tuple/list block output must contain "
                    "(output, ephemeral_state)."
                )
            out, ephemeral_state = out

        h_next = out.squeeze(1)

        if return_ephemeral:
            return h_next, ephemeral_state

        return h_next

    def candidate_successors(
        self,
        h,
        h0,
        memory,
        context,
        pointer,
    ):
        """
        Generate independent successor states for block-selection actions.

        Ordering:
            0 ... N-1 : ADVANCE(0 ... N-1)
            N         : REPEAT
            N+1       : RESTART

        HALT intentionally has no successor and is therefore excluded
        from this successor tensor.
        """
        if h.ndim != 2:
            raise ValueError("h must have shape [B, D].")

        if h0.shape != h.shape:
            raise ValueError("h0 shape must match h shape.")

        if memory.shape != h.shape:
            raise ValueError("memory shape must match h shape.")

        num_blocks = len(self.block_bank.blocks)

        if not 0 <= pointer < num_blocks:
            raise ValueError(
                f"pointer must be in [0, {num_blocks - 1}], "
                f"got {pointer}."
            )

        candidate_h = []
        next_pointer = []

        # ADVANCE(j') for every candidate block.
        for block_index in range(num_blocks):
            h_next = self._run_block(
                block_index=block_index,
                h=h,
                h0=h0,
                memory=memory,
                context=context,
            )
            candidate_h.append(h_next)
            next_pointer.append(block_index)

        # REPEAT current block.
        h_repeat = self._run_block(
            block_index=pointer,
            h=h,
            h0=h0,
            memory=memory,
            context=context,
        )
        candidate_h.append(h_repeat)
        next_pointer.append(pointer)

        # RESTART is zero-compute and preserves h.
        candidate_h.append(h)
        next_pointer.append(0)

        return {
            "h": torch.stack(candidate_h, dim=1),
            "next_pointer": next_pointer,
        }



    def differentiable_route_step(
        self,
        h,
        h0,
        memory,
        context,
        pointer,
        action_logits,
        tau,
        training=True,
        execution_count=0,
        just_restarted=False,
    ):
        """
        Execute one differentiable Gate-C routing decision.

        The candidate successor set is:

            ADVANCE(0 ... N-1), REPEAT, RESTART

        HALT is intentionally excluded because it has no successor state.

        Forward:
            ST-Gumbel produces a hard one-hot action and therefore selects
            exactly one candidate successor.

        Backward:
            the straight-through relaxed routing path carries gradients
            through action_logits and the candidate successor computation.

        No persistent cache is modified by this method.
        """
        if h.ndim != 2:
            raise ValueError("h must have shape [B, D].")

        if h0.shape != h.shape:
            raise ValueError("h0 shape must match h shape.")

        if memory.shape != h.shape:
            raise ValueError("memory shape must match h shape.")

        if action_logits.ndim != 2:
            raise ValueError(
                "action_logits must have shape [B, N+3]."
            )

        num_blocks = len(self.block_bank.blocks)
        expected_actions = num_blocks + 3

        if action_logits.shape[-1] != expected_actions:
            raise ValueError(
                f"action_logits must have {expected_actions} actions "
                f"for N={num_blocks}; got {action_logits.shape[-1]}."
            )

        if action_logits.shape[0] != h.shape[0]:
            raise ValueError(
                "action_logits batch size must match h batch size."
            )

        # --------------------------------------------------------
        # Candidate successor states.
        #
        # candidate_successors() returns N+2 states:
        #   ADVANCE[*], REPEAT, RESTART
        #
        # HALT has no successor and is excluded.
        # --------------------------------------------------------
        candidates = self.candidate_successors(
            h=h,
            h0=h0,
            memory=memory,
            context=context,
            pointer=pointer,
        )

        candidate_h = candidates["h"]

        # --------------------------------------------------------
        # Build the legality mask using the existing executor
        # contract, then remove HALT from the successor space.
        # --------------------------------------------------------
        full_mask = build_execution_mask(
            pointer=pointer,
            execution_count=execution_count,
            just_restarted=just_restarted,
            num_blocks=num_blocks,
        ).to(device=action_logits.device)

        successor_count = num_blocks + 2

        successor_mask = full_mask[:successor_count]
        successor_mask = successor_mask.unsqueeze(0).expand(
            action_logits.shape[0],
            -1,
        )

        # --------------------------------------------------------
        # HALT remains a controller action but has no successor.
        # Therefore it is not passed to successor mixing.
        # --------------------------------------------------------
        block_selection_logits = action_logits[:, :successor_count]

        h_next, action, probs = st_gumbel_successor_mix(
            candidates=candidate_h,
            logits=block_selection_logits,
            mask=successor_mask,
            tau=tau,
            training=training,
        )

        # --------------------------------------------------------
        # Pointer transition is discrete metadata. It follows the
        # hard forward action and is intentionally not differentiated.
        # --------------------------------------------------------
        selected_action = action.detach().argmax(dim=-1)

        next_pointer_table = torch.tensor(
            candidates["next_pointer"],
            device=action.device,
            dtype=torch.long,
        )

        next_pointer = next_pointer_table[selected_action]

        return {
            "h_next": h_next,
            "action": action,
            "probs": probs,
            "candidates": candidate_h,
            "selected_action": selected_action,
            "next_pointer": next_pointer,
        }


    def execute_differentiable(
        self,
        h0,
        memory,
        C_prev=None,
        context=None,
        memory_mean=None,
        budget=None,
        tau=1.0,
        training=True,
    ):
        """
        Fixed-depth differentiable Gate-C executor.

        This path intentionally does not replace the legacy `execute()`
        method. It provides the Stage-3 training/integration path:

            controller
                -> masked action logits
                -> ST-Gumbel successor routing
                -> hard pointer transition
                -> next controller decision

        HALT is excluded from successor mixing because it has no successor
        state. Execution therefore continues for exactly
        `max_execution_depth` routing decisions.

        Current pilot contract:
            - batch size = 1
            - controller uses the canonical Stage-3 interface
            - pointer transition is hard/discrete metadata
            - persistent context is never mutated
            - no convergence gate
            - no learned HALT signal
        """
        if h0.ndim != 2:
            raise ValueError("h0 must have shape [B, D].")

        if memory.ndim != 2 or memory.shape != h0.shape:
            raise ValueError(
                "memory must have the same shape as h0: [B, D]."
            )

        batch_size = h0.shape[0]

        # The current executor candidate construction has one shared
        # pointer, so the differentiable pilot remains batch-size one.
        if batch_size != 1:
            raise ValueError(
                "execute_differentiable currently requires batch size 1."
            )

        num_blocks = len(self.block_bank.blocks)

        if memory_mean is None:
            memory_mean = torch.zeros(
                batch_size,
                256,
                device=h0.device,
                dtype=h0.dtype,
            )
        else:
            if memory_mean.ndim != 2 or memory_mean.shape != (batch_size, 256):
                raise ValueError(
                    "memory_mean must have shape [B, 256]."
                )

        if budget is None:
            budget = torch.zeros(
                batch_size,
                2,
                device=h0.device,
                dtype=h0.dtype,
            )
        else:
            if budget.ndim != 2 or budget.shape != (batch_size, 2):
                raise ValueError(
                    "budget must have shape [B, 2]."
                )

        h = h0.clone()
        pointer = 0
        just_restarted = False
        action_history = []

        for execution_count in range(self.max_execution_depth):
            pointer_one_hot = torch.zeros(
                batch_size,
                num_blocks + 1,
                device=h.device,
                dtype=h.dtype,
            )
            pointer_one_hot[:, pointer] = 1.0

            controller_out = self.controller(
                h=h,
                memory_read=memory,
                memory_mean=memory_mean,
                pointer_one_hot=pointer_one_hot,
                budget=budget,
            )

            action_logits = controller_out["logits"]

            if training:
                # Training path: exhaustive successor construction is required
                # so the ST-Gumbel relaxed backward path can differentiate
                # through every legal successor.
                result = self.differentiable_route_step(
                    h=h,
                    h0=h0,
                    memory=memory,
                    context=context,
                    pointer=pointer,
                    action_logits=action_logits,
                    tau=tau,
                    training=True,
                    execution_count=execution_count,
                    just_restarted=just_restarted,
                )

                h = result["h_next"]
                selected_action = int(
                    result["selected_action"][0].item()
                )

            else:
                # Deterministic inference path: only the selected legal
                # successor is executed. Exhaustive candidate construction is
                # unnecessary when no routing gradient is required.
                full_mask = build_execution_mask(
                    pointer=pointer,
                    execution_count=execution_count,
                    just_restarted=just_restarted,
                    num_blocks=num_blocks,
                ).to(device=action_logits.device)

                # Fixed-depth Stage-3 inference has no learned HALT
                # successor. Therefore deterministic routing selects only
                # from the N+2 successor actions:
                #   ADVANCE(0..N-1), REPEAT, RESTART
                # HALT is intentionally excluded.
                successor_count = num_blocks + 2
                successor_mask = full_mask[:successor_count]

                masked_logits = apply_action_mask(
                    action_logits[:, :successor_count],
                    successor_mask.unsqueeze(0),
                )

                selected_action = int(
                    torch.argmax(
                        masked_logits,
                        dim=-1,
                    )[0].item()
                )

                if selected_action < num_blocks:
                    h = self._run_block(
                        block_index=selected_action,
                        h=h,
                        h0=h0,
                        memory=memory,
                        context=context,
                    )
                elif selected_action == num_blocks:
                    h = self._run_block(
                        block_index=pointer,
                        h=h,
                        h0=h0,
                        memory=memory,
                        context=context,
                    )
                else:
                    # RESTART is zero-compute and preserves h.
                    h = h

            if selected_action < num_blocks:
                pointer = selected_action
                just_restarted = False
                action_history.append(
                    ("ADVANCE", pointer)
                )
            elif selected_action == num_blocks:
                just_restarted = False
                action_history.append(
                    ("REPEAT", pointer)
                )
            else:
                pointer = 0
                just_restarted = True
                action_history.append(
                    ("RESTART", None)
                )

        return ExecutionResult(
            h_terminal=h,
            execution_count=self.max_execution_depth,
            action_history=action_history,
            halted=False,
            safety_cap_hit=False,
        )

    def execute(
        self,
        h0,
        memory,
        C_prev,
        context=None,
        memory_mean=None,
        budget=None,
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

        batch_size = h.shape[0]

        # Gate-C controller inputs.
        #
        # memory_mean and budget are explicit inputs. No budget schedule
        # is invented here. When omitted by legacy callers, zero tensors
        # preserve backward compatibility.
        if memory_mean is None:
            memory_mean = torch.zeros(
                batch_size,
                256,
                device=h.device,
                dtype=h.dtype,
            )
        else:
            if memory_mean.ndim != 2 or memory_mean.shape != (batch_size, 256):
                raise ValueError(
                    "memory_mean must have shape [B, 256]."
                )

        if budget is None:
            budget = torch.zeros(
                batch_size,
                2,
                device=h.device,
                dtype=h.dtype,
            )
        else:
            if budget.ndim != 2 or budget.shape != (batch_size, 2):
                raise ValueError(
                    "budget must have shape [B, 2]."
                )

        action_history = []

        halted = False
        safety_cap_hit = False

        frozen_memory = memory
        frozen_C_prev = C_prev

        del frozen_C_prev  # retained only as an explicit compatibility input

        while execution_count < self.max_execution_depth:

            pointer_one_hot = torch.zeros(
                batch_size,
                len(self.block_bank.blocks) + 1,
                device=h.device,
                dtype=h.dtype,
            )
            pointer_one_hot[:, pointer] = 1.0

            # Prefer the canonical Stage-3 controller interface.
            # Legacy controllers remain supported until the executor
            # itself is migrated to differentiable routing.
            try:
                target = getattr(self.controller, "forward", self.controller)
                signature = inspect.signature(target)
                canonical_controller = all(
                    name in signature.parameters
                    for name in (
                        "h",
                        "memory_read",
                        "memory_mean",
                        "pointer_one_hot",
                        "budget",
                    )
                )
            except (AttributeError, TypeError, ValueError):
                canonical_controller = False

            if canonical_controller:
                controller_out = self.controller(
                    h=h,
                    memory_read=memory,
                    memory_mean=memory_mean,
                    pointer_one_hot=pointer_one_hot,
                    budget=budget,
                )
            else:
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
