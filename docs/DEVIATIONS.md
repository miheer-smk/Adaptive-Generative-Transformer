# Deviations from Blueprint

None yet.

- D6 (resolved for the bank): block bank GDN now uses src/gdn_v3.py (GDNv3), built from blueprint 7.6 (mixer = 2,374,796 params at d=768). src/gdn.py (legacy) is NOT modified and is no longer used by the bank; two GDN implementations exist until tests/test_gdn.py is migrated. Remove legacy after migration.

- D13: repo apparently has two 768-param final norms (TerminalCommit and output_head); blueprint has one RMSNorm_f. terminal_commit takes a single injected final_norm; reconcile before wiring so that l_{t,P} = W_out s_t holds.
- M2.4: src/terminal_commit.py MemoryWrite duplicates the existing PersistentMemory.write semantics; reconcile when memory.py is wired (existing commit.py / memory.py / prelude.py were not modified).
