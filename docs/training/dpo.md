# Direct preference optimization (DPO)

The third of three training phases. See [`pretrain.md`](pretrain.md)
and [`sft.md`](sft.md) for what comes before.

## What it does

Given a base + an SFT adapter, take it one step further: train the
model to PREFER one response over another. The data is
`{prompt, chosen, rejected}` triplets — humans or a stronger model
ranked the two responses.

## Math

Define the implicit reward function:
```
r_θ(y | x) = log π_θ(y | x) - log π_ref(y | x)
```

where `π_θ` is the policy (the model we're training) and `π_ref` is a
frozen reference (a copy of the base before DPO). Then the DPO loss:

```
L_DPO = - E_{(x, y_w, y_l)} [ log σ ( β · (r_θ(y_w | x) - r_θ(y_l | x)) ) ]
```

Expanding the reward:

```
L_DPO = - log σ ( β · ( logπ_pol(chosen)   - logπ_pol(rejected)
                      - logπ_ref(chosen)   + logπ_ref(rejected) ) )
```

At step 0, `π_θ = π_ref` (policy starts as a copy of reference), so the
log-ratios cancel and the inner expression is 0; the loss is
`-log σ(0) = log 2 ≈ 0.693`. **That's the canonical sanity check —
the first DPO step should print loss ≈ 0.69.**

`β` is the temperature: lower keeps the policy close to the reference
(safer, more conservative); higher sharpens preferences (more
aggressive, more risk of drift). 0.1 is a typical default.

## Why a reference model?

The reference is a regularizer. Without it, the model would maximize
chosen-vs-rejected by any means including catastrophic
shifts in the output distribution. The KL constraint to the reference
keeps the policy in a meaningful neighborhood of the base.

Memory cost: ~2× the base size (policy + reference both held in
memory). At bf16 on a 100M Mega, that's ~400 MB.

## What datasets to use

| Dataset | Size | Source of preference | Notes |
|---|---:|---|---|
| `HuggingFaceH4/ultrafeedback_binarized` | 60K pairs | GPT-4 judgments | Strong default. |
| `argilla/dpo-mix-7k` | 7K | mixed sources, cleaned | Smaller, higher per-example quality |
| `anthropic/hh-rlhf` | ~170K | human labels | Slow but human-grade |

Full catalog with URLs and licenses in
[`docs/roadmap/datasets.md`](../roadmap/datasets.md).

## Reproduce

```bash
# Once we tokenize UltraFeedback into the JSONL shape DPO expects.
.xcode-build/Build/Products/Debug/tinygpt dpo \
    /tmp/mega-fineweb.tinygpt \
    --data /tmp/ultrafeedback.jsonl \
    --template chatml \
    --rank 4 --alpha 8 \
    --beta 0.1 \
    --steps 500 \
    --lr 5e-5 \
    --out /tmp/mega-dpo.lora
```

`tinygpt dpo` accepts either the flat `{prompt, chosen, rejected}`
shape or the HF chat-array shape — see `PreferenceReader` for details.

## How to know it worked

DPO loss alone is hard to interpret directly. The useful signal is
**preference accuracy**: at evaluation, sample two responses from the
policy and the reference for the same held-out prompt, run them through
a stronger judge model, and report what fraction of the time the policy
beats the reference. That's an upcoming `tinygpt dpo-eval` command;
for now, eyeball samples.

## Background reading

- DPO: Rafailov et al., 2023 ("[Direct Preference Optimization: Your
  Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)"),
  NeurIPS 2023. The closed-form derivation in §4 is the math we implement.
- Other preference recipes — SimPO, ORPO, KTO, IPO — in
  [`docs/PLAN.md`](../PLAN.md) §4.1 (Alignment / preference).
