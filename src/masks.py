
import torch


def build_execution_mask(
    pointer: int,
    execution_count: int,
    just_restarted: bool,
    num_blocks: int,
):
    """
    Build the legal-action mask for one execution state.

    Unified controller actions:

        ADVANCE(j') : 0 ... N-1
        REPEAT      : N
        RESTART     : N+1
        HALT        : N+2

    The mask is boolean:
        True  = legal
        False = illegal

    Contract:
    - pointer identifies the current block position.
    - execution_count tracks actual executions.
    - just_restarted prevents consecutive RESTART actions.
    """

    if not 0 <= pointer < num_blocks:
        raise ValueError(
            f"pointer must be in [0, {num_blocks - 1}], "
            f"got {pointer}"
        )

    if execution_count < 0:
        raise ValueError(
            f"execution_count must be >= 0, "
            f"got {execution_count}"
        )

    num_actions = num_blocks + 3

    mask = torch.ones(
        num_actions,
        dtype=torch.bool,
    )

    # ---------------------------------------------------------
    # ADVANCE(j')
    # ---------------------------------------------------------
    # Cannot advance beyond the final block.
    mask[:num_blocks] = True

    # ---------------------------------------------------------
    # REPEAT
    # ---------------------------------------------------------
    repeat_index = num_blocks
    mask[repeat_index] = True

    # ---------------------------------------------------------
    # RESTART
    # ---------------------------------------------------------
    restart_index = num_blocks + 1

    # No consecutive RESTART actions.
    if just_restarted:
        mask[restart_index] = False

    # ---------------------------------------------------------
    # HALT
    # ---------------------------------------------------------
    halt_index = num_blocks + 2
    mask[halt_index] = True

    return mask


def apply_action_mask(
    logits: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Apply a boolean legality mask to controller logits.

    Illegal actions receive -inf.
    """

    if logits.shape[-1] != mask.shape[-1]:
        raise ValueError(
            "logits and mask must have the same action dimension"
        )

    return logits.masked_fill(
        ~mask.to(device=logits.device),
        float("-inf"),
    )
