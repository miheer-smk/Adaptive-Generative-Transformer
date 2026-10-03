
import torch




def build_execution_mask(
    pointer: int,
    execution_count: int,
    just_restarted: bool,
    num_blocks: int,
):
    """
    Exact source-aligned action legality mask.

    Internal block indices are 0-based:
        block 0 == source block j=1
        block N-1 == source block j=N

    Action layout:
        0 ... N-1 : ADVANCE(j')
        N         : REPEAT
        N+1       : RESTART
        N+2       : HALT

    States:

    Initial state:
        pointer=0, execution_count=0
        all ADVANCE(0..N-1) allowed
        REPEAT masked
        RESTART masked
        HALT allowed

    After executing block i within a sweep:
        pointer=i, execution_count>0
        ADVANCE(i+1..N-1) allowed
        REPEAT allowed
        RESTART allowed unless immediately after RESTART
        HALT allowed

    Immediately after RESTART:
        pointer=0, just_restarted=True
        all ADVANCE(0..N-1) allowed
        REPEAT masked
        RESTART masked
        HALT allowed
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

    mask = torch.zeros(
        num_actions,
        dtype=torch.bool,
    )

    # ---------------------------------------------------------
    # ADVANCE(j')
    # ---------------------------------------------------------
    # Initial state and post-RESTART state are represented by
    # pointer=0 but differ from having already executed block 0.
    if execution_count == 0 or just_restarted:
        mask[:num_blocks] = True
    elif pointer < num_blocks - 1:
        mask[pointer + 1:num_blocks] = True

    # ---------------------------------------------------------
    # REPEAT
    # ---------------------------------------------------------
    repeat_index = num_blocks

    if execution_count > 0 and not just_restarted:
        mask[repeat_index] = True

    # ---------------------------------------------------------
    # RESTART
    # ---------------------------------------------------------
    restart_index = num_blocks + 1

    if execution_count > 0 and not just_restarted:
        mask[restart_index] = True

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
