# Implementation Decisions

Format per entry:
Decision ID / Date / Component / Blueprint reference / Status /
Chosen implementation / Reason / Alternative considered / Risk /
Must be revisited? / Experiment or unit test

Source of truth: AGT-v3 blueprint pre-freeze v0.1 (30 Sep 2026).

Index convention: Python token index i <-> mathematical t = i+1.

## GDN ephemeral/commit milestone (src/gdn_v3.py)
- GDN-1 (PROVISIONAL, blueprint 7.6 silent): A_h = exp(A_log) with A_log init 0 (A_h = 1) so alpha in (0,1). Revisit: Stage-0 stability, learned-A drift.
- GDN-2 (PROVISIONAL): b_h init 0.
- GDN-3 (PROVISIONAL): q/k/v/g/w_alpha/w_beta init N(0, 0.02); W_o init N(0, 0.02/sqrt(2N)), N=8 (blueprint 8 residual-scaling rule applied to GDN W_o).
- GDN-4 (PROVISIONAL): conv weights at framework default init; no conv bias.
- GDN-5: conv buffer = last (kernel-1)=3 PRE-conv [q|k|v] projections, [B,3,1536], zero-initialised when no committed context.
- GDN-6: gates alpha/beta computed from the block input u (pre-conv), per blueprint.
- GDN-7: AGTBlock.gdn_commit_entry computes the commit entry from u_c = RMSNorm(h_terminal + a*h0 + m*R_c) (I4); pure, caller persists via GDNStateCache.commit_terminal_context.

## M2.4 terminal commit (src/terminal_commit.py, AGTBlock.commit_entry)
- AMB-11 (PROVISIONAL, blueprint 4.2/15.1 ambiguous): the prelude block's committed entry is computed from the prelude block's OWN input u_t (the entry its ephemeral self attention already used); the prelude has no terminal-state input. Needs owner confirmation.
- CMT-1: the caller supplies R_c = Read(h_terminal, M_t) and K_M^t (key projection computed once per token by the memory read). The existing memory read must expose K_M.
- CMT-2 (PROVISIONAL): MemoryWrite is a new spec-faithful module (394,241 params at 768/256); Ww_q, Ww_v, w_g init N(0, 0.02); b_g = -2. Existing PersistentMemory.write is not used.
- CMT-3: one injected final_norm provides s_t (blueprint: single RMSNorm_f shared with the readout).
- CMT-4: commit is two-phase (pure build_commit_plan, single apply_commit_plan) with state-consumed and per-cache version guards.
- CMT-5: cache keys are bank index i (0-based; math block j = i+1) and 'pre'; all caches created eagerly in AGTPersistentState.initial.
