import torch
import torch.nn.functional as F


def st_gumbel_action(
    logits: torch.Tensor,
    mask: torch.Tensor,
    tau: float,
    training: bool = True,
):
    """
    Straight-through Gumbel-Softmax categorical action estimator.

    Args:
        logits:
            Action logits with final dimension = number of actions.

        mask:
            Boolean legality mask with the same shape as logits.
            True  -> legal action
            False -> illegal action

        tau:
            Positive Gumbel-Softmax temperature.

        training:
            True  -> stochastic ST-Gumbel-Softmax.
            False -> deterministic argmax over masked logits.

    Returns:
        action:
            Straight-through action tensor.
            Forward value is exactly one-hot over legal actions.

        probs:
            Masked categorical probabilities pi used by the controller.
            Illegal actions have probability exactly zero.

    Contract:
        pi = softmax(masked_logits)

        g ~ Gumbel(0, 1)

        z = softmax((log(pi) + g) / tau)

        forward:
            a = one_hot(argmax(z))

        backward:
            a_tilde = a + z - stop_gradient(z)
    """
    if not torch.is_tensor(logits):
        raise TypeError("logits must be a torch.Tensor.")

    if not torch.is_tensor(mask):
        raise TypeError("mask must be a torch.Tensor.")

    if logits.shape != mask.shape:
        raise ValueError(
            "logits and mask must have identical shapes."
        )

    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension.")

    if not isinstance(tau, (int, float)):
        raise TypeError("tau must be a positive scalar.")

    if tau <= 0:
        raise ValueError("tau must be > 0.")

    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)

    # Every categorical row must contain at least one legal action.
    legal_count = mask.sum(dim=-1)
    if torch.any(legal_count == 0):
        raise ValueError(
            "Every categorical row must have at least one legal action."
        )

    # Illegal actions are removed before probability normalization.
    masked_logits = logits.masked_fill(
        ~mask,
        float("-inf"),
    )

    probs = F.softmax(masked_logits, dim=-1)

    if not training:
        # Inference is deterministic: argmax over pi.
        indices = torch.argmax(masked_logits, dim=-1)
        action = F.one_hot(
            indices,
            num_classes=logits.shape[-1],
        ).to(dtype=logits.dtype)

        return action, probs

    # Sample standard Gumbel noise.
    uniform = torch.rand_like(logits)
    eps = torch.finfo(logits.dtype).eps
    uniform = uniform.clamp(min=eps, max=1.0 - eps)

    gumbel = -torch.log(
        -torch.log(uniform)
    )

    # Use log(pi) as specified by the ST-Gumbel formulation.
    # Invalid actions remain structurally invalid.
    log_probs = torch.where(
        mask,
        torch.log(probs.clamp_min(eps)),
        torch.zeros_like(probs),
    )

    relaxed = F.softmax(
        (log_probs + gumbel) / tau,
        dim=-1,
    )

    # Enforce exact zero for illegal actions.
    relaxed = relaxed.masked_fill(
        ~mask,
        0.0,
    )

    # Hard categorical forward pass.
    indices = torch.argmax(relaxed, dim=-1)
    hard = F.one_hot(
        indices,
        num_classes=logits.shape[-1],
    ).to(dtype=logits.dtype)

    # Straight-through estimator:
    # forward = hard
    # backward = relaxed
    action = hard - relaxed.detach() + relaxed

    return action, probs


def st_gumbel_successor_mix(
    candidates: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor,
    tau: float,
    training: bool = True,
):
    """
    Route among candidate successor states using the ST-Gumbel action.

    Args:
        candidates:
            Tensor shaped [B, A, ...], where A is the action count.

        logits:
            Action logits shaped [B, A].

        mask:
            Boolean legality mask shaped [B, A].

        tau:
            Positive Gumbel-Softmax temperature.

        training:
            True  -> stochastic straight-through Gumbel routing.
            False -> deterministic argmax routing.

    Returns:
        mixed:
            Candidate successor mixture shaped [B, ...].

        action:
            Straight-through one-hot action tensor [B, A].

        probs:
            Masked categorical probabilities [B, A].

    Forward semantics:
        mixed = selected candidate successor exactly.

    Backward semantics:
        gradients flow through the relaxed categorical action.
    """
    if candidates.ndim < 2:
        raise ValueError(
            "candidates must have shape [B, A, ...]."
        )

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape [B, A]."
        )

    if mask.shape != logits.shape:
        raise ValueError(
            "mask and logits must have identical shapes."
        )

    if candidates.shape[:2] != logits.shape:
        raise ValueError(
            "candidates first two dimensions must match logits [B, A]."
        )

    action, probs = st_gumbel_action(
        logits=logits,
        mask=mask,
        tau=tau,
        training=training,
    )

    weights = action

    # Broadcast the [B, A] routing weights across every successor-state
    # dimension.
    while weights.ndim < candidates.ndim:
        weights = weights.unsqueeze(-1)

    mixed = (weights * candidates).sum(dim=1)

    return mixed, action, probs
