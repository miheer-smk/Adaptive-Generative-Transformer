"""
AGT-v3 Gated DeltaNet (blueprint 7.6) with committed / ephemeral context.

Per head (H heads, key dim dk, value dim dv):
    S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T,   o_t = S_t q_t
    alpha_t = exp(-A_h * softplus(w_alpha^T u + b_h))   in (0, 1)
    beta_t  = sigmoid(w_beta^T u)                       in (0, 1)
q, k, v come from a short causal depthwise conv (kernel 4, SiLU) over the
concatenated projections [q | k | v]; q, k are L2-normalised.
Output: RMSNorm_head(o_t) * SiLU(W_g u), then W_o.

AGT context semantics
---------------------
Committed context C_{t-1} for this block = (S_{t-1}, conv_buffer_{t-1}), where
conv_buffer is the last (kernel-1) PRE-conv [q|k|v] inputs, shape [B, K-1, C].

forward_ephemeral(u, state, conv_buffer)  -> GDNStep(y, S_tilde, new_buffer)
    PURE. Reads the committed context, never writes it. S_tilde / new_buffer
    are temporaries that the caller discards (in-flight execution).
commit(u_c, state, conv_buffer)           -> (S_t, conv_buffer_t)
    PURE. Same recurrence evaluated on the terminal-state input u_c. The
    caller persists the result exactly once via
    GDNStateCache.commit_terminal_context.
Nothing in this module mutates a cache or its tensors.
"""
from typing import NamedTuple, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import RMSNorm


class GDNStep(NamedTuple):
    y: torch.Tensor            # [B, T, dim]
    state: torch.Tensor        # [B, H, dv, dk]   ephemeral S_tilde
    conv_buffer: torch.Tensor  # [B, K-1, C]      ephemeral buffer


class GDNv3(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        dk: int = 64,
        dv: int = 128,
        conv_kernel: int = 4,
        residual_scale_blocks: int = 8,
    ):
        super().__init__()
        if conv_kernel < 2:
            raise ValueError("conv_kernel must be >= 2.")

        self.dim = dim
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv
        self.conv_kernel = conv_kernel
        self.q_dim = num_heads * dk
        self.v_dim = num_heads * dv
        self.conv_channels = 2 * self.q_dim + self.v_dim

        self.q_proj = nn.Linear(dim, self.q_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.q_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.v_dim, bias=False)
        self.g_proj = nn.Linear(dim, self.v_dim, bias=False)
        self.o_proj = nn.Linear(self.v_dim, dim, bias=False)
        self.w_alpha = nn.Linear(dim, num_heads, bias=False)
        self.w_beta = nn.Linear(dim, num_heads, bias=False)

        # A_h = exp(A_log) > 0 keeps alpha in (0, 1). Provisional (GDN-1).
        self.A_log = nn.Parameter(torch.zeros(num_heads))
        self.b_h = nn.Parameter(torch.zeros(num_heads))

        # Depthwise causal conv; no padding (history comes from the buffer).
        self.conv = nn.Conv1d(
            self.conv_channels, self.conv_channels, conv_kernel,
            groups=self.conv_channels, bias=False,
        )
        self.head_norm = RMSNorm(dv)

        for lin in (self.q_proj, self.k_proj, self.v_proj,
                    self.g_proj, self.w_alpha, self.w_beta):
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
        nn.init.normal_(
            self.o_proj.weight, mean=0.0,
            std=0.02 / math.sqrt(2 * residual_scale_blocks),
        )

    # ------------------------------------------------------------------
    def _gates(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """u: [B, T, dim] -> alpha, beta: [B, T, H]."""
        A = torch.exp(self.A_log)
        alpha = torch.exp(-A * F.softplus(self.w_alpha(u) + self.b_h))
        beta = torch.sigmoid(self.w_beta(u))
        return alpha, beta

    @staticmethod
    def _step(S, q, k, v, alpha, beta):
        """One delta-rule step. S [B,H,dv,dk]; q,k [B,H,dk]; v [B,H,dv];
        alpha, beta [B,H]. Returns (S_new, o) with o [B,H,dv]."""
        a = alpha[..., None, None]
        b = beta[..., None, None]
        Sk = torch.einsum("bhvk,bhk->bhv", S, k)
        S_new = a * (S - b * Sk.unsqueeze(-1) * k.unsqueeze(-2)) \
            + b * v.unsqueeze(-1) * k.unsqueeze(-2)
        o = torch.einsum("bhvk,bhk->bhv", S_new, q)
        return S_new, o

    def _run(self, u, state, conv_buffer, compute_output):
        if u.ndim != 3 or u.shape[-1] != self.dim:
            raise ValueError(f"GDN input must be [B, T, {self.dim}].")
        B, T, _ = u.shape
        if T < 1:
            raise ValueError("Input sequence length must be positive.")
        H, dk, dv = self.num_heads, self.dk, self.dv
        K1 = self.conv_kernel - 1

        pre = torch.cat(
            [self.q_proj(u), self.k_proj(u), self.v_proj(u)], dim=-1
        )  # [B, T, C]

        if conv_buffer is None:
            buf = pre.new_zeros(B, K1, self.conv_channels)
        else:
            if tuple(conv_buffer.shape) != (B, K1, self.conv_channels):
                raise ValueError(
                    f"conv_buffer shape {tuple(conv_buffer.shape)} != "
                    f"{(B, K1, self.conv_channels)}."
                )
            buf = conv_buffer

        seq = torch.cat([buf, pre], dim=1)                    # [B, K1+T, C]
        new_buf = seq[:, -K1:, :].clone()                     # last K-1 pre-conv inputs
        conv_out = F.silu(self.conv(seq.transpose(1, 2)).transpose(1, 2))  # [B, T, C]

        q, k, v = conv_out.split([self.q_dim, self.q_dim, self.v_dim], dim=-1)
        q = F.normalize(q.reshape(B, T, H, dk), dim=-1)
        k = F.normalize(k.reshape(B, T, H, dk), dim=-1)
        v = v.reshape(B, T, H, dv)
        alpha, beta = self._gates(u)

        if state is None:
            S = pre.new_zeros(B, H, dv, dk)
        else:
            if tuple(state.shape) != (B, H, dv, dk):
                raise ValueError(
                    f"state shape {tuple(state.shape)} != {(B, H, dv, dk)}."
                )
            S = state

        outs = []
        for t in range(T):
            S, o = self._step(S, q[:, t], k[:, t], v[:, t], alpha[:, t], beta[:, t])
            if compute_output:
                outs.append(o)

        y = None
        if compute_output:
            o = self.head_norm(torch.stack(outs, dim=1))      # [B, T, H, dv]
            g = F.silu(self.g_proj(u)).reshape(B, T, H, dv)
            y = self.o_proj((o * g).reshape(B, T, H * dv))

        return y, S, new_buf

    # ------------------------------------------------------------------
    def forward_ephemeral(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        conv_buffer: Optional[torch.Tensor] = None,
    ) -> GDNStep:
        """PURE in-flight execution. Reads committed (state, conv_buffer)."""
        y, S, buf = self._run(u, state, conv_buffer, True)
        return GDNStep(y, S, buf)

    def commit(
        self,
        u_c: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        conv_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PURE. Terminal entry (S_t, conv_buffer_t) from the terminal-state input."""
        _, S, buf = self._run(u_c, state, conv_buffer, False)
        return S, buf

    def forward(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        conv_buffer: Optional[torch.Tensor] = None,
    ):
        """Block-interface form: (y, ephemeral_state). Never writes a cache."""
        step = self.forward_ephemeral(u, state=state, conv_buffer=conv_buffer)
        return step.y, step.state
