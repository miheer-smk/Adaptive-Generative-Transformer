
import torch


class KVCache:
    """
    Simple append-only KV cache for causal attention.

    Cache stores post-norm, post-RoPE K/V as specified by
    provisional AMB-3.
    """

    def __init__(self):
        self.keys = None
        self.values = None

    def clear(self):
        self.keys = None
        self.values = None

    def append(self, keys, values):
        """
        keys, values:
            [B, H, T, D]

        Returns the complete cached K/V tensors.
        """
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = torch.cat(
                [self.keys, keys],
                dim=2,
            )
            self.values = torch.cat(
                [self.values, values],
                dim=2,
            )

        return self.keys, self.values

    def get(self):
        return self.keys, self.values

    @property
    def length(self):
        if self.keys is None:
            return 0
        return self.keys.shape[2]


class GDNStateCache:
    """
    Explicit recurrent-state cache for GDN blocks.

    The state is maintained separately from token activations.
    """

    def __init__(self):
        self.state = None

    def clear(self):
        self.state = None

    def set(self, state):
        self.state = state

    def get(self):
        return self.state


class ExecutionCache:
    """
    Container for per-execution transient caches.

    Persistent H/M/C state remains outside this object.
    """

    def __init__(self):
        self.kv = {}
        self.gdn = {}

    def clear(self):
        self.kv.clear()
        self.gdn.clear()

    def get_kv_cache(self, block_id):
        if block_id not in self.kv:
            self.kv[block_id] = KVCache()
        return self.kv[block_id]

    def get_gdn_cache(self, block_id):
        if block_id not in self.gdn:
            self.gdn[block_id] = GDNStateCache()
        return self.gdn[block_id]
