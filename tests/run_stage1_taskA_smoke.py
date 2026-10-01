import torch
import torch.nn.functional as F

from src.agt import AGTv3


torch.manual_seed(3001)
torch.set_num_threads(4)

DIM = 64
VOCAB = 32
SLOTS = 16
BATCH = 16
SEQ_LEN = 8
STEPS = 120

# Fixed synthetic token representation.
g = torch.Generator().manual_seed(3002)
token_features = torch.randn(VOCAB, DIM, generator=g)


def make_batch():
    tokens = torch.randint(
        low=0,
        high=VOCAB,
        size=(BATCH, SEQ_LEN),
    )

    x = token_features[tokens]
    targets = (tokens + 1) % VOCAB

    return x, targets


def grad_report(model):
    print("\n=== GRADIENT COVERAGE ===")

    required = {
        "memory": 0,
        "prelude": 0,
        "blocks": 0,
        "commit": 0,
        "memory_write": 0,
        "output_head": 0,
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            if param.grad.abs().sum().item() > 0:
                top = name.split(".", 1)[0]
                if top in required:
                    required[top] += 1

    for name, count in required.items():
        print(f"{name:15s}: {count:4d} tensors with nonzero gradient")

    missing = [name for name, count in required.items() if count == 0]

    assert not missing, f"Missing gradient coverage: {missing}"


def main():
    model = AGTv3(
        dim=DIM,
        num_memory_slots=SLOTS,
        vocab_size=VOCAB,
    ).train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    print("=== STAGE 1 FIXED-ROUTE CE SMOKE TEST ===")
    print(f"dim={DIM}, vocab={VOCAB}, batch={BATCH}, seq_len={SEQ_LEN}, steps={STEPS}")

    first_loss = None
    last_loss = None

    for step in range(1, STEPS + 1):
        x, targets = make_batch()

        state = model.initial_persistent_state(batch_size=BATCH)
        C_prev = torch.zeros(BATCH, DIM)

        logits_per_t = []

        for t in range(SEQ_LEN):
            out = model.step(
                x[:, t, :],
                state,
                C_prev,
            )

            logits_per_t.append(out["logits"])
            state = out["state"]

        logits = torch.stack(logits_per_t, dim=1)

        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB),
            targets.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if step == 1:
            first_loss = loss.item()
            grad_report(model)

        optimizer.step()

        last_loss = loss.item()

        if step == 1 or step % 20 == 0:
            print(f"step={step:03d}  loss={loss.item():.6f}")

    print("\n=== RESULT ===")
    print(f"initial_loss = {first_loss:.6f}")
    print(f"final_loss   = {last_loss:.6f}")

    assert torch.isfinite(torch.tensor(first_loss))
    assert torch.isfinite(torch.tensor(last_loss))

    # Basic trainability signal; exact benchmark thresholds belong to
    # the real Task-A specification, which is not present in this repo.
    assert last_loss < first_loss, (
        f"Loss did not decrease: {first_loss:.6f} -> {last_loss:.6f}"
    )

    print("STAGE 1 SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
