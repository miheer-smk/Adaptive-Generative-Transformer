# Editable Mermaid diagrams

The SVG/PNG fallbacks and these Mermaid sources share one editable graph definition. Static images use the coordinates in diagrams.json; Mermaid chooses its own layout. Arrows do not assert scientific validation.

## AGT-v3: implemented token computation and persistent state

D64 research pilots; task adapters vary; fixed-execution HALT disabled

```mermaid
flowchart LR
    input["Input / task encoder"]
    state["Incoming committed state<br/>s, M, typed context C<br/>Pilot memory: 16 x 256"]
    prepare["Clone working state<br/>Read(s_previous, M)<br/>R reused within token"]
    prelude["Prelude -> h0<br/>Read-only token anchor"]
    route["Execution path<br/>Canonical / scripted / controller<br/>Chosen by experiment"]
    bank["Heterogeneous bank<br/>SWA x2 -> GQA x2<br/>-> GDN x2 -> FFN x2"]
    context["Working typed context<br/>KV / SWA windows<br/>GDN state + conv buffers"]
    commit["Terminal commit<br/>Normalize stream; write memory<br/>Commit all contextual entries"]
    output["Task readout<br/>Parity projection /<br/>CLRS dynamic pointer head"]
    next["Returned committed state<br/>s_new, M_new, C_new<br/>Carry to next token"]
    input --> prelude
    state --> prepare
    prepare --> prelude
    prelude --> route
    route --> bank
    prepare -->|"h0 / R"| bank
    prepare -.->|"cloned context"| context
    context -.->|"read only"| bank
    bank -->|"h_terminal"| commit
    context -.->|"plan inputs"| commit
    commit -->|"s_new"| output
    commit -->|"once only"| next
```

## One token: live execution ends at a single commit boundary

Corrected fixed-execution path; P=8 is a pilot setting, D counts decisions

```mermaid
flowchart LR
    in["Incoming state<br/>Parent stays unchanged"]
    prep["Owned clone<br/>Memory read + Prelude"]
    act["Legal action loop<br/>ADVANCE / REPEAT: D+1, P+1<br/>RESTART: D+1, P+0<br/>RESTART preserves live h"]
    term["Terminal live state<br/>Reached actual P budget"]
    plan["Pure terminal plan<br/>Validate ownership / versions"]
    return["Apply once<br/>Return new committed state"]
    cache["Committed KV / GDN / window context<br/>Read-only during all in-flight block calls"]
    in --> prep
    prep --> act
    act -->|"P = 8 (pilot)"| term
    term --> plan
    plan --> return
    prep -.->|"cloned"| cache
    cache -.->|"read only"| act
    cache -.->|"terminal context"| plan
```

## Research lineage: questions and evidence, not a chain of successes

Dashed arrows mean next question / reused substrate; exact statuses remain in the index

```mermaid
flowchart LR
    mechanics["Mechanics + parity reference<br/>Scoped positive controls"]
    imitation["Canonical imitation / replay<br/>Supervised recovery"]
    discovery["Task-loss routing discovery<br/>Not established"]
    programs["Complete-program feasibility<br/>v1-v3 stopped; v4 rejected"]
    receiver["Receiver-anchored parity<br/>COMPLETED_NON_SUCCESS"]
    keyed["Keyed-register qualification<br/>Resource stop; unqualified"]
    clrs["CLRS fixed-program learning<br/>COMPLETED_NON_SUCCESS"]
    audit["Localization / residual / mHC<br/>Bounded diagnostics"]
    probe["Frozen bilinear readouts<br/>Uncertified then certified<br/>Competence not recovered"]
    feature["Feature-complement paired pilot<br/>PARTIAL_EFFECT_TASK_NON_SUCCESS"]
    next["No executed successor found<br/>New research decision required"]
    mechanics -.-> imitation
    imitation -.-> discovery
    discovery -.-> programs
    programs -.-> receiver
    receiver -.-> keyed
    keyed -.-> clrs
    clrs -.-> audit
    audit -.-> probe
    probe -.-> feature
    feature -.-> next
```

## Serialized official insertion-sort predecessor task

Fresh D64 AGT + task adapter; not the official CLRS GNN; fixed canonical execution

```mermaid
flowchart LR
    array["Official array<br/>Original keys + positions"]
    inputs["First n input tokens<br/>Type flags, position, key<br/>Candidate index = original j"]
    queries["Next n query tokens<br/>Type flags, position i<br/>No key or answer feature"]
    head["Dynamic pointer head<br/>score(i,j) = Q_i dot K_j / 8<br/>All n candidates"]
    candidates["Save committed candidate z_j<br/>After input element j<br/>Autograd-connected in training"]
    core["Same AGT step, 2n calls<br/>8 selected blocks per token<br/>Normal state carry + commits"]
    loss["Query-only mean pointer CE<br/>Official original-index targets<br/>Minimum self-pointer; stable ties"]
    array --> inputs
    inputs -->|"serialization order"| queries
    inputs -->|"input calls"| core
    queries -->|"query calls"| core
    core -->|"input states"| candidates
    candidates -->|"projected keys"| head
    core -->|"projected queries"| head
    head -->|"predictions"| loss
    array -.->|"targets: supervision only"| loss
```
