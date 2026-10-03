import inspect
import torch
import torch.nn as nn

from .norms import RMSNorm
from .gqa import GQA
from .gdn_v3 import GDNv3 as GDN, GDNStep
from .mhc import MHC
from .swiglu import SwiGLU
from .rope import apply_rope
from .caches import GDNStateCache, KVCache, SWARingCache


class SlidingWindowAttention(GQA):
    """
    SWA species: GQA projections/RoPE with a sliding window of W positions.

    No cache (training-style full sequence):
        position i attends to j with 0 <= i - j < W.

    With kv_cache (AGT per-token protocol, T must be 1):
        visible context = (<= W-1 committed rows) + (1 ephemeral self entry)
        = at most W positions, all causal and in-window by construction.
        The cache is NOT mutated. RoPE uses the absolute token position
        (kv_cache.next_position), not the retained-row count.
    """

    def __init__(
        self,
        dim: int,
        num_query_heads: int = 8,
        num_kv_heads: int = 2,
        window_size: int = 256,
        max_seq_len: int = 256,
    ):
        super().__init__(
            dim=dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
        )

        self.window_size = window_size

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("SWA input must have shape [B, T, dim].")

        B, T, _ = x.shape
        W = self.window_size

        if kv_cache is None:
            q, k, v = self._project_qkv_rope(x, 0)
            k, v = self._repeat_kv_heads(k, v)

            pos = torch.arange(T, device=x.device)
            dist = pos[:, None] - pos[None, :]          # i - j
            allowed = (dist >= 0) & (dist < W)

            attn_mask = torch.zeros(
                (T, T), device=x.device, dtype=q.dtype
            ).masked_fill(~allowed, float("-inf"))

            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            if not isinstance(kv_cache, SWARingCache):
                raise TypeError("SWA requires an SWARingCache.")
            if kv_cache.window_size != W:
                raise ValueError(
                    f"Cache window_size={kv_cache.window_size} != "
                    f"SWA window_size={W}."
                )
            if T != 1:
                raise ValueError(
                    "Cached SWA supports exactly one token per call "
                    "(AGT per-token protocol)."
                )

            position = kv_cache.next_position
            q, k, v = self._project_qkv_rope(x, position)

            # Pure: concatenates committed rows with the local self K/V.
            k_ctx, v_ctx = kv_cache.make_ephemeral_context(k, v)

            if k_ctx.shape[2] > W:
                raise RuntimeError(
                    f"SWA visible context {k_ctx.shape[2]} exceeds W={W}."
                )

            k_ctx, v_ctx = self._repeat_kv_heads(k_ctx, v_ctx)

            # All visible keys are earlier committed tokens inside the
            # window, plus self: no mask is required.
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k_ctx, v_ctx,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

        y = y.transpose(1, 2).contiguous().view(
            B, T, self.num_query_heads * self.head_dim
        )

        return self.o_proj(y)


class FFNHeavyCore(nn.Module):
    """
    Attention-free FFN-heavy block core.

    Block-interface milestone: accepts an optional additive `injection`
    (a_j * h0 + m_j * R, computed by AGTBlock). The injection enters the
    sub-layer INPUT only, before the pre-norm. The residual path is the raw x.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = RMSNorm(dim)
        self.ffn = SwiGLU(
            dim=dim,
            hidden_dim=hidden_dim,
        )

        self.mhc = MHC(dim=dim)

    def forward(
        self,
        x: torch.Tensor,
        injection: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x

        pre = x if injection is None else x + injection

        h = self.norm(pre)
        h = self.ffn(h)
        h = self.mhc(h)

        return residual + h


_GDN_STATE_KW = ("state", "initial_state", "prev_state", "past_state", "S_prev")
_GDN_CONV_KW = ("conv_buffer", "conv_state", "prev_conv", "conv_cache")


class AGTBlock(nn.Module):
    """
    Single heterogeneous AGT-v3 block, executor-facing form:

        F_j(h, h0, R, C_prev)

    Blueprint sections 5.2 / 15 (AMB-1 confirmed):
        u   = RMSNorm( h + a_j * h0 + m_j * R )      # sub-layer input only
        out = h + mhc( core(u; C_prev) )              # residual is the RAW h

    (Streams are broadcast from a single h and H_pre is a softmax, so
    H_pre . r == h exactly; the injected form above is the AMB-1 form.)

    Arguments:
        h      [B, T, d]  live state h_{t,p}
        h0     [B, T, d]  read-only anchor (None -> no anchor injection)
        R      [B, T, d]  memory read R(h, M_t) (None -> no memory injection)
        C_prev            this block's COMMITTED context (never mutated here):
                              GQA -> KVCache
                              SWA -> SWARingCache
                              GDN -> GDNStateCache
                              FFN -> None

    With h0=None and R=None the block reduces to the pre-interface behavior.

    Returns:
        GQA / SWA / FFN : tensor [B, T, d]
        GDN             : (tensor [B, T, d], ephemeral_state). The state is NOT
                          written to C_prev; only terminal commit may do that.

    Supported species: SWA, GQA, GDN, FFN
    """

    def __init__(
        self,
        dim: int = 768,
        species: str = "GQA",
        swa_window: int = 256,
        ffn_hidden_dim: int = 3072,
        max_seq_len: int = 256,
    ):
        super().__init__()

        self.dim = dim
        self.species = species.upper()

        self.norm = RMSNorm(dim)

        if self.species == "GQA":
            self.core = GQA(
                dim=dim,
                num_query_heads=8,
                num_kv_heads=2,
                max_seq_len=max_seq_len,
            )
            self.mhc = MHC(dim=dim)

        elif self.species == "SWA":
            self.core = SlidingWindowAttention(
                dim=dim,
                num_query_heads=8,
                num_kv_heads=2,
                window_size=swa_window,
                max_seq_len=max_seq_len,
            )
            self.mhc = MHC(dim=dim)

        elif self.species == "GDN":
            self.core = GDN(dim=dim)
            self.mhc = MHC(dim=dim)

        elif self.species == "FFN":
            self.core = FFNHeavyCore(
                dim=dim,
                hidden_dim=ffn_hidden_dim,
            )
            self.mhc = None

        else:
            raise ValueError(
                f"Unknown AGT block species: {species}"
            )

        # Step-independent diagonal injection gates (blueprint 5.2, init 0.1).
        self.anchor_gate = nn.Parameter(torch.full((dim,), 0.1))
        self.memory_gate = nn.Parameter(torch.full((dim,), 0.1))

    # ------------------------------------------------------------------
    def injection(self, h, h0=None, R=None):
        """a_j * h0 + m_j * R (None if both are None)."""
        inj = None

        if h0 is not None:
            if h0.shape != h.shape:
                raise ValueError(
                    f"h0 shape {tuple(h0.shape)} != h shape {tuple(h.shape)}."
                )
            inj = self.anchor_gate * h0

        if R is not None:
            if R.shape != h.shape:
                raise ValueError(
                    f"R shape {tuple(R.shape)} != h shape {tuple(h.shape)}."
                )
            term = self.memory_gate * R
            inj = term if inj is None else inj + term

        return inj

    def _gdn_context_kwargs(self, C_prev):
        """Map the committed GDN context onto the core's keyword arguments."""
        if C_prev is None:
            return {}

        if not isinstance(C_prev, GDNStateCache):
            raise TypeError("GDN block requires a GDNStateCache as C_prev.")

        state, conv = C_prev.get_committed_context()
        sig = inspect.signature(self.core.forward)
        params = sig.parameters

        out = {}
        for value, names, what in (
            (state, _GDN_STATE_KW, "state"),
            (conv, _GDN_CONV_KW, "convolution buffer"),
        ):
            if value is None:
                continue
            name = next((n for n in names if n in params), None)
            if name is None:
                raise NotImplementedError(
                    f"GDN core.forward{sig} has no parameter that accepts the "
                    f"committed {what} (looked for {names}). The GDN "
                    f"ephemeral/commit milestone must define this interface."
                )
            out[name] = value

        return out

    # ------------------------------------------------------------------
    def gdn_commit_entry(self, h_terminal, h0=None, R_c=None, C_prev=None):
        """
        PURE. The GDN terminal-commit entry for this block (blueprint 15.1 ii):

            u_c = RMSNorm( h_terminal + a_j*h0 + m_j*R_c )      # same u as in-flight
            (S_t, conv_buffer_t) = core.commit(u_c; committed S_{t-1}, buffer)

        Returns (state, conv_buffer). Does NOT write C_prev; the dedicated
        terminal commit persists it once via GDNStateCache.commit_terminal_context.
        Valid for blocks that were never visited this token (C_prev may be empty).
        """
        if self.species != "GDN":
            raise ValueError("gdn_commit_entry is only defined for GDN blocks.")
        if C_prev is not None and not isinstance(C_prev, GDNStateCache):
            raise TypeError("GDN block requires a GDNStateCache as C_prev.")

        inj = self.injection(h_terminal, h0, R_c)
        pre = h_terminal if inj is None else h_terminal + inj
        u_c = self.norm(pre)

        if C_prev is None:
            state, conv = None, None
        else:
            state, conv = C_prev.get_committed_context()

        return self.core.commit(u_c, state=state, conv_buffer=conv)

    def attn_commit_entry(self, h_terminal, h0=None, R_c=None, C_prev=None):
        """
        PURE. SWA/GQA terminal-commit entry (blueprint 15.1 ii, invariant I4):

            u_c = RMSNorm( h_terminal + a_j*h0 + m_j*R_c )      # same u as in-flight
            (k, v) = project_kv(u_c, absolute position = C_prev.next_position)

        k is QK-normed and RoPE-rotated (the cached form). Returns (k, v),
        each [B, Hkv, 1, D]. Does NOT write C_prev.
        """
        if self.species not in {"GQA", "SWA"}:
            raise ValueError("attn_commit_entry is only defined for GQA/SWA blocks.")
        if C_prev is None:
            raise ValueError(
                "The committed cache is required: it defines the absolute position."
            )
        if self.species == "SWA":
            if not isinstance(C_prev, SWARingCache):
                raise TypeError("SWA block requires an SWARingCache as C_prev.")
            if C_prev.window_size != self.core.window_size:
                raise ValueError(
                    f"Cache window_size={C_prev.window_size} != "
                    f"SWA window_size={self.core.window_size}."
                )
        else:
            if not isinstance(C_prev, KVCache) or isinstance(C_prev, SWARingCache):
                raise TypeError("GQA block requires a plain KVCache as C_prev.")
        if h_terminal.ndim != 3 or h_terminal.shape[1] != 1:
            raise ValueError("A commit entry is defined for exactly one token [B, 1, d].")

        inj = self.injection(h_terminal, h0, R_c)
        pre = h_terminal if inj is None else h_terminal + inj
        u_c = self.norm(pre)
        return self.core.project_kv(u_c, C_prev.next_position)

    def commit_entry(self, h_terminal, h0=None, R_c=None, C_prev=None):
        """
        PURE. Species-typed terminal-commit entry (never writes C_prev):
            GQA / SWA : (k, v)
            GDN       : (state, conv_buffer)
            FFN       : None (empty context)
        """
        if self.species in {"GQA", "SWA"}:
            return self.attn_commit_entry(h_terminal, h0, R_c, C_prev)
        if self.species == "GDN":
            return self.gdn_commit_entry(h_terminal, h0, R_c, C_prev)
        if self.species == "FFN":
            if C_prev is not None:
                raise ValueError("FFN-heavy has an empty context; C_prev must be None.")
            return None
        raise RuntimeError(f"Unsupported species: {self.species}")

    def forward(self, h, h0=None, R=None, C_prev=None, **kwargs):
        return_ephemeral = bool(kwargs.pop("return_ephemeral", False))

        inj = self.injection(h, h0, R)

        if self.species == "FFN":
            if C_prev is not None:
                raise ValueError(
                    "FFN-heavy has an empty context; C_prev must be None."
                )
            return self.core(h, injection=inj)

        pre = h if inj is None else h + inj
        u = self.norm(pre)

        if self.species in {"GQA", "SWA"}:
            if C_prev is not None:
                if "kv_cache" in kwargs:
                    raise ValueError(
                        "Pass the context via C_prev or kv_cache, not both."
                    )
                if self.species == "SWA":
                    if not isinstance(C_prev, SWARingCache):
                        raise TypeError("SWA block requires an SWARingCache.")
                else:
                    if (
                        not isinstance(C_prev, KVCache)
                        or isinstance(C_prev, SWARingCache)
                    ):
                        raise TypeError("GQA block requires a plain KVCache.")
                kwargs = {**kwargs, "kv_cache": C_prev}

            y = self.core(u, **kwargs)
            y = self.mhc(y)
            return h + y

        if self.species == "GDN":
            call_kw = {**kwargs, **self._gdn_context_kwargs(C_prev)}

            if return_ephemeral:
                # Opt-in Gate-C path: preserve the complete in-flight
                # GDN transition, including recurrent state and
                # convolution buffer. The persistent cache is untouched.
                step = self.core.forward_ephemeral(u, **call_kw)

                if not isinstance(step, GDNStep):
                    raise RuntimeError(
                        "GDN forward_ephemeral() must return GDNStep."
                    )

                y = self.mhc(step.y)
                return h + y, step

            out = self.core(u, **call_kw)

            if not (isinstance(out, tuple) and len(out) == 2):
                raise RuntimeError(
                    "GDN core must return (output, state)."
                )

            y, state = out
            y = self.mhc(y)
            return h + y, state

        raise RuntimeError(
            f"Unsupported species: {self.species}"
        )


class HeterogeneousBlockBank(nn.Module):
    """
    AGT-v3 heterogeneous block bank.

    Required composition:
        2 SWA
        2 GQA
        2 GDN
        2 FFN-heavy
    """

    def __init__(
        self,
        dim: int = 768,
        species_order=None,
        swa_window: int = 256,
        ffn_hidden_dim: int = 3072,
        max_seq_len: int = 256,
    ):
        super().__init__()

        if species_order is None:
            species_order = [
                "SWA",
                "SWA",
                "GQA",
                "GQA",
                "GDN",
                "GDN",
                "FFN",
                "FFN",
            ]

        if len(species_order) != 8:
            raise ValueError(
                "AGT-v3 requires exactly 8 blocks."
            )

        self.species_order = [
            species.upper()
            for species in species_order
        ]

        expected = {
            "SWA": 2,
            "GQA": 2,
            "GDN": 2,
            "FFN": 2,
        }

        actual = {
            species: self.species_order.count(species)
            for species in expected
        }

        if actual != expected:
            raise ValueError(
                f"Invalid AGT-v3 bank composition: {actual}. "
                f"Expected: {expected}."
            )

        self.blocks = nn.ModuleList(
            [
                AGTBlock(
                    dim=dim,
                    species=species,
                    swa_window=swa_window,
                    ffn_hidden_dim=ffn_hidden_dim,
                    max_seq_len=max_seq_len,
                )
                for species in self.species_order
            ]
        )

    def __len__(self):
        return len(self.blocks)

    def forward(self, index, x, **kwargs):
        return self.blocks[index](
            x,
            **kwargs,
        )