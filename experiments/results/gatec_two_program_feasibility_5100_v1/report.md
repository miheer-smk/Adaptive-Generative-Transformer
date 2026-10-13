# Two-program feasibility: STOPPED

{
  "error": "RuntimeError('expected self and mask to be on the same device, but got mask on cpu and self on cuda:0')",
  "traceback": "Traceback (most recent call last):\n  File \"/home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC/experiments/complete_program_feasibility.py\", line 623, in execute\n    b,sb,br=runner.run(x,state_adapter,P0,label=label,case=dict(gate='archival',arm='adapter',token=token))\n            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/iiitn/agt-v3-env/lib/python3.12/site-packages/torch/utils/_contextlib.py\", line 124, in decorate_context\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC/experiments/complete_program_feasibility.py\", line 348, in run\n    out = model.adaptive_differentiable_step(x, parent, compatibility,\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC/src/agt.py\", line 399, in adaptive_differentiable_step\n    execution = self.executor.execute_differentiable(\n                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC/src/executor.py\", line 647, in execute_differentiable\n    mask = build_execution_mask(\n           ^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC/experiments/complete_program_feasibility.py\", line 229, in mask\n    selected = int(row['logits'][:, :10].masked_fill(~result[:10], -torch.inf).argmax(-1))\n                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nRuntimeError: expected self and mask to be on the same device, but got mask on cpu and self on cuda:0\n",
  "first_mechanical": {
    "case": {
      "gate": "archival",
      "arm": "adapter",
      "token": 0
    },
    "token": 0,
    "error": "RuntimeError('expected self and mask to be on the same device, but got mask on cpu and self on cuda:0')",
    "observed_actions": [
      null
    ],
    "observed_block_calls": [
      []
    ]
  },
  "completed_tokens": 1,
  "elapsed_seconds": 0.9332003137096763
}

No training, selector or follow-up acquisition was run.
