# v2 inspection stop: no repair or model execution

The v1 attempt remains immutable and STOPPED BEFORE P1. v2 stopped during the required inspection.

## Confirmed device-only root cause

The sealed observer at `v1/complete_program_feasibility.py:224-241` obtains the source mask and applies its first ten boolean entries to CUDA FP32 logits before returning it. `src/masks.py:65-68` constructs a CPU bool mask of shape11. Scripted logits inherit FP32/CUDA from h and have shape1x11. The retained v1 failure confirms the device mismatch at line229.

Production at `src/executor.py:647-652` normally moves the returned mask to `action_logits.device`. That placement happens after the patched observer returns and therefore cannot protect the observer's earlier local operation. The contemplated correction transfers only the local mask operand to the logits device, retaining shape,boolean dtype,values,legal indices,action ordering,-inf masking and argmax. Neither input is modified in place, and no RNG is needed. This correction was not applied.

## Separate stopping/gate conflict

The sealed runner records a current-task pair at lines672-674 and enters future schedules at679-690 unconditionally. Additional errors,pointwise CE/TV violations and absolute correctness are inspected only at726-731 after every planned continuation. This cannot satisfy the current authorization's prohibition on later stages whose prerequisites failed. Fixing gate control would be a separate semantic change to the research runner, outside the one permitted observer device correction.

## Archival comparison completeness

Lines624-626 compare new reference and adapter full states and route metadata. Lines627-631 compare retained archival logits and intermediate h values. The retained fixture also contains terminal,commit/commit_pre,context summaries and discrete metadata; these historical records are not all explicitly compared in that archival gate. Current runtime commit checks are not themselves historical-record comparisons. Providing the requested complete archival comparison would require another change beyond observer device placement.

## Execution boundary

No versioned corrected source or tests were created; no test runs,CUDA validation,model forwards/backwards,training,optimizer,selector or Git commit occurred. v2 counts are zero. v1's one completed reference token,8 bank calls and one commit remain separate historical partial work,not v2 outcome evidence.

Engineering: root cause confirmed; repair validation blocked by the inspection conflicts. Current-task correctness: unverified. Future compatibility: unverified. Research hypothesis: untested. Neither candidate program was accepted or rejected scientifically.

The observer-only authorization cannot produce a compliant full study from this runner unchanged. No broader correction was improvised.
