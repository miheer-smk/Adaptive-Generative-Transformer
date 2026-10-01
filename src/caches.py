import torch


class KVCache:
    """
    AGT-v3 committed KV context.

    M2.1 invariant:
        In-flight execution is pure with respect to persistent cache state.

    Therefore:

        get_committed_context()
            -> read-only committed context

        make_ephemeral_context(keys, values)
            -> construct visible context for the current in-flight call
            -> DOES NOT mutate the cache

        commit_terminal_context(keys, values)
            -> persist newly produced K/V
            -> increments version exactly once

    Shapes:
        keys, values: [B, H, T, D]
    """

    def __init__(self):
        self.committed_keys = None
        self.committed_values = None

        # Version changes only when persistent committed state changes.
        self.version = 0

    def clear(self):
        self.committed_keys = None
        self.committed_values = None
        self.version = 0

    def get_committed_context(self):
        """
        Return only committed context.

        This operation is read-only.
        """
        return (
            self.committed_keys,
            self.committed_values,
        )

    def make_ephemeral_context(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """
        Construct:

            committed context + supplied ephemeral K/V

        without modifying persistent cache state.
        """

        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError(
                "keys and values must have shape [B, H, T, D]."
            )

        if keys.shape != values.shape:
            raise ValueError(
                "keys and values must have identical shapes."
            )

        if self.committed_keys is None:
            return keys, values

        visible_keys = torch.cat(
            [
                self.committed_keys,
                keys,
            ],
            dim=2,
        )

        visible_values = torch.cat(
            [
                self.committed_values,
                values,
            ],
            dim=2,
        )

        return visible_keys, visible_values

    def commit_terminal_context(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """
        Persist newly produced K/V.

        This is the ONLY operation here that changes committed context.
        """

        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError(
                "keys and values must have shape [B, H, T, D]."
            )

        if keys.shape != values.shape:
            raise ValueError(
                "keys and values must have identical shapes."
            )

        if self.committed_keys is None:
            self.committed_keys = keys
            self.committed_values = values
        else:
            self.committed_keys = torch.cat(
                [
                    self.committed_keys,
                    keys,
                ],
                dim=2,
            )

            self.committed_values = torch.cat(
                [
                    self.committed_values,
                    values,
                ],
                dim=2,
            )

        self.version += 1

    # ---------------------------------------------------------
    # Compatibility helpers
    # ---------------------------------------------------------

    def append_ephemeral(self, keys, values):
        """
        Compatibility wrapper.

        IMPORTANT:
        This no longer mutates cache state.
        It simply constructs the visible context.
        """
        return self.make_ephemeral_context(
            keys,
            values,
        )

    def append(self, keys, values):
        """
        Backward-compatible persistent append.

        New AGT code should prefer:
            commit_terminal_context()
        """
        self.commit_terminal_context(
            keys,
            values,
        )

        return self.get_committed_context()

    def get(self):
        """
        Backward-compatible read.

        Returns committed context only.
        """
        return self.get_committed_context()

    @property
    def committed_length(self):
        if self.committed_keys is None:
            return 0

        return self.committed_keys.shape[2]

    @property
    def ephemeral_length(self):
        """
        Ephemeral state is not stored in the cache.

        Therefore this is always zero.
        """
        return 0

    @property
    def next_position(self):
        """Absolute token position of the next token to be committed.
        For an unbounded KV cache this equals the number of committed rows."""
        return self.committed_length

    @property
    def length(self):
        return self.committed_length


    def clone(self):
        """
        Return an independent persistent-cache snapshot.

        Tensor storage is cloned and the original cache is untouched.
        """
        out = KVCache()
        out.committed_keys = (
            None if self.committed_keys is None
            else self.committed_keys.clone()
        )
        out.committed_values = (
            None if self.committed_values is None
            else self.committed_values.clone()
        )
        out.version = self.version
        return out


class SWARingCache(KVCache):
    """
    Sliding-window attention cache.

    window_size = W = total attention window = (committed rows) + (1 ephemeral
    self entry). The cache therefore retains the most recent W-1 committed
    rows. Absolute token position is tracked separately from the number of
    retained rows, because RoPE uses absolute position.

    Truncation is by slicing the most recent rows (chronological order is
    always preserved); semantics are identical to a circular buffer.

    In-flight execution is pure (inherited from KVCache). Only
    commit_terminal_context() changes state, once per call.
    """

    def __init__(self, window_size=128):
        if window_size < 2:
            raise ValueError(
                "window_size must be >= 2 (W-1 committed rows + self)."
            )

        super().__init__()

        self.window_size = window_size
        self.num_committed_tokens = 0

    @property
    def retained_capacity(self):
        return self.window_size - 1

    def clear(self):
        super().clear()
        self.num_committed_tokens = 0

    @property
    def next_position(self):
        return self.num_committed_tokens

    def commit_terminal_context(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """
        Persist K/V, advance absolute position, retain only W-1 rows.
        """

        super().commit_terminal_context(
            keys,
            values,
        )

        self.num_committed_tokens += keys.shape[2]

        cap = self.window_size - 1

        if self.committed_keys.shape[2] > cap:
            self.committed_keys = self.committed_keys[:, :, -cap:, :]
            self.committed_values = self.committed_values[:, :, -cap:, :]

    def commit(self):
        """
        Compatibility no-op retained for older call sites.
        """
        return


    def clone(self):
        """
        Return an independent persistent SWA cache snapshot.
        """
        out = SWARingCache(window_size=self.window_size)
        out.committed_keys = (
            None if self.committed_keys is None
            else self.committed_keys.clone()
        )
        out.committed_values = (
            None if self.committed_values is None
            else self.committed_values.clone()
        )
        out.version = self.version
        out.num_committed_tokens = self.num_committed_tokens
        return out


class GDNStateCache:
    """
    Committed GDN recurrent state.

    M2.1 invariant:
        In-flight state is supplied to/from the block call and
        is not written into persistent cache until terminal commit.

    The convolution buffer fields are retained for M2.3.
    """

    def __init__(self):
        self.committed_state = None
        self.committed_conv_buffer = None

        self.version = 0

    def clear(self):
        self.committed_state = None
        self.committed_conv_buffer = None
        self.version = 0

    def get_committed_context(self):
        """
        Return committed recurrent state and convolution buffer.
        """
        return (
            self.committed_state,
            self.committed_conv_buffer,
        )

    def make_ephemeral_context(
        self,
        state,
        conv_buffer=None,
    ):
        """
        Return supplied in-flight state without mutating cache.
        """

        return state, conv_buffer

    def commit_terminal_context(
        self,
        state,
        conv_buffer=None,
    ):
        """
        Persist terminal GDN state and optional convolution buffer.
        """

        self.committed_state = state
        self.committed_conv_buffer = conv_buffer
        self.version += 1

    # ---------------------------------------------------------
    # Compatibility helpers
    # ---------------------------------------------------------

    def set_ephemeral(
        self,
        state,
        conv_buffer=None,
    ):
        """
        Compatibility helper.

        Does NOT mutate persistent state.
        """
        return self.make_ephemeral_context(
            state,
            conv_buffer,
        )

    def set(
        self,
        state,
        conv_buffer=None,
    ):
        """
        Backward-compatible non-mutating setter.
        """
        return self.make_ephemeral_context(
            state,
            conv_buffer,
        )

    def get(self):
        return self.get_committed_context()

    def commit(self):
        """
        Retained for compatibility.

        Terminal callers should use commit_terminal_context()
        with explicit state.
        """
        return


    def clone(self):
        """
        Return an independent persistent GDN cache snapshot.
        """
        out = GDNStateCache()
        out.committed_state = (
            None if self.committed_state is None
            else self.committed_state.clone()
        )
        out.committed_conv_buffer = (
            None if self.committed_conv_buffer is None
            else self.committed_conv_buffer.clone()
        )
        out.version = self.version
        return out


class ExecutionCache:
    """
    Per-execution cache container.

    Persistent model state H/M remains outside this object.

    Cache types:
        GQA -> KVCache
        SWA -> SWARingCache
        GDN -> GDNStateCache
    """

    def __init__(self, swa_window=128):
        self.swa_window = swa_window

        self.kv = {}
        self.swa = {}
        self.gdn = {}

    def clear(self):
        for cache in self.kv.values():
            cache.clear()

        for cache in self.swa.values():
            cache.clear()

        for cache in self.gdn.values():
            cache.clear()

        self.kv.clear()
        self.swa.clear()
        self.gdn.clear()

    def get_kv_cache(self, block_id):
        if block_id not in self.kv:
            self.kv[block_id] = KVCache()

        return self.kv[block_id]

    def get_swa_cache(self, block_id):
        if block_id not in self.swa:
            self.swa[block_id] = SWARingCache(
                window_size=self.swa_window
            )

        return self.swa[block_id]

    def get_gdn_cache(self, block_id):
        if block_id not in self.gdn:
            self.gdn[block_id] = GDNStateCache()

        return self.gdn[block_id]
    
    def clone(self):
        """
        Return an independent snapshot of every committed cache.
        """
        out = ExecutionCache(swa_window=self.swa_window)

        out.kv = {
            key: cache.clone()
            for key, cache in self.kv.items()
        }
        out.swa = {
            key: cache.clone()
            for key, cache in self.swa.items()
        }
        out.gdn = {
            key: cache.clone()
            for key, cache in self.gdn.items()
        }

        return out
