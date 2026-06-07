# External references — articles worth reading

Curated articles, papers, and projects relevant to TinyGPT. Each entry
is one-sentence-what + one-sentence-why-for-us + link, per the docs
preference for leaning on authoritative external sources rather than
re-explaining them.

Updated 2026-06-08.

---

## LLM internals — pedagogical explainers

- **[The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)**
  Visual explainer of the original transformer; got a 2025 refresh with
  animations.
  *Why for us*: the bar for "explain a transformer with diagrams" — any
  TinyGPT learning doc should link here rather than redraw the diagrams.

- **[Transformer Explainer (Poloclub)](https://poloclub.github.io/transformer-explainer/)**
  Interactive GPT-2 running live in the browser; click any layer to see
  values flow.
  *Why for us*: pair with our forward-pass walkthrough sessions — lets
  readers poke a real model after reading our annotated code.

- **[Lil'Log (Lilian Weng)](https://lilianweng.github.io/)**
  Long-form deep technical essays on inference, training, evals, and
  agents.
  *Why for us*: the quality bar for technical writing we should aim for
  in TinyGPT's own learning docs.

- **[Lil'Log — Why We Think (May 2025)](https://lilianweng.github.io/posts/2025-05-01-thinking/)**
  40-min read on test-time compute and why small models + better
  inference can beat scaling up.
  *Why for us*: direct intellectual backing for the TinyGPT thesis
  (specialists + on-device > frontier APIs).

- **[Karpathy — 2025 LLM Year in Review](https://karpathy.bearblog.dev/year-in-review-2025/)**
  Opinionated short post on the paradigm shifts of the year.
  *Why for us*: useful framing for the HN-launch positioning of
  TinyGPT relative to where the field is.

- **[ByteByteGo — How Transformers Architecture Powers Modern LLMs](https://blog.bytebytego.com/p/how-transformers-architecture-powers)**
  Diagram-heavy walkthrough of the seven-step decode loop.
  *Why for us*: closest reference for what a TinyGPT explainer post
  should look like structurally.

- **[Stephen Wolfram — What Is ChatGPT Doing](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)**
  First-principles essay on next-token prediction without assuming any
  ML background.
  *Why for us*: the bar for accessibility; useful model for our
  `learn.md` entry-point.

- **[Maxleiter — They're Made Out of Weights](https://maxleiter.com/blog/weights)**
  Terry Bisson "Made Out of Meat" parody — short dialogue framing LLMs
  as numbers all the way down.
  *Why for us*: example of literary/vibe-shaped writing about LLMs;
  inspiration for a Sarthak voice piece on specialists.

- **[Terry Bisson — They're Made Out of Meat (original)](https://www.mit.edu/people/dpolicar/writing/prose/text/thinkingMeat.html)**
  The 1991 sci-fi dialogue Leiter's piece riffs on.
  *Why for us*: read first to get the joke.

- **[Janelle Shane — AI Weirdness](https://www.aiweirdness.com/)**
  Long-running humor/AI internals blog (e.g., what neural nets name
  things weirdly).
  *Why for us*: closest practicing tradition for whimsy + ML mechanics;
  good rhythm reference.

---

## Apple Neural Engine / Core ML / Mac-native LLMs

(Critical reading for the ANE arc — see `docs/learn/ane-research/dossier.md`
for our synthesized dossier of these and adjacent sources.)

- **[Apple ML Research — Deploying Transformers on the Apple Neural Engine (2022)](https://machinelearning.apple.com/research/neural-engine-transformers)**
  The canonical paper introducing the `(B, C, 1, S)` layout and
  ane_gelu / ane_silu / LayerNormANE patterns.
  *Why for us*: the reference our ANE M7 layout port works from.

- **[Apple ML Research — On-Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama)**
  Apple's own walkthrough showing Llama-3.1-8B at ~33 tok/s on M1 Max
  via Core ML.
  *Why for us*: direct precedent for the Qwen3 ANE conversion we're
  attempting; their KV-cache pattern matters.

- **[HuggingFace — Running Mistral 7B with Core ML (WWDC '24)](https://huggingface.co/blog/mistral-coreml)**
  Step-by-step conversion of Mistral 7B to .mlpackage via coremltools.
  *Why for us*: the cleanest community walkthrough; pairs with our
  `scripts/ane/qwen3_to_coreml.py`.

- **[HuggingFace — Releasing Swift Transformers](https://huggingface.co/blog/swift-coreml-llm)**
  Swift-side LLM inference plumbing.
  *Why for us*: parallel work to TinyGPTServe — worth checking for
  things they've already solved that we shouldn't re-implement.

- **[CoreML-LLM (john-rocky, GitHub)](https://github.com/john-rocky/CoreML-LLM)**
  Community project hitting 52 tok/s on ANE for Gemma 4 / Qwen3.5 /
  Qwen3-VL with zero GPU contention.
  *Why for us*: closest direct competitor to TinyGPT's ANE work;
  understand their patterns before shipping our own.

- **[Orion — Characterizing and Programming Apple's Neural Engine for LLM Training and Inference (arXiv)](https://arxiv.org/html/2603.06728v1)**
  Recent reverse-engineering paper on programming the ANE directly.
  *Why for us*: cited in our M8 research arcs; useful for going beyond
  what coremltools exposes.

- **[BrightCoding — Stop Wasting GPU Cycles: CoreML-LLM Unlocks ANE](https://www.blog.brightcoding.dev/2026/05/23/stop-wasting-gpu-cycles-coreml-llm-unlocks-ane-for-insane-on-device-speed)**
  Community post benchmarking the CoreML-LLM project on ANE.
  *Why for us*: real-world tok/s numbers on the same silicon we
  target; sanity check for our benchmarks.

---

## Distillation + specialist models

- **[Predibase — 12 Best Practices for Distilling Small LMs from GPT](https://predibase.com/blog/graduate-from-openai-to-open-source-12-best-practices-for-distilling-smaller)**
  Practitioner checklist for teacher→student distillation at production
  scale.
  *Why for us*: closest analog to the Pace specialist arc; cross-check
  our methodology against their twelve points.

- **[Labelbox — End-to-end distillation with Gemini](https://labelbox.com/guides/end-to-end-workflow-for-knowledge-distillation-with-nlp/)**
  Concrete walkthrough: label with teacher, train student, compare.
  *Why for us*: structure to borrow for the TinyGPT "factory" docs.

- **[Nebius — The Concept Behind Distilling an LLM](https://nebius.com/blog/posts/concept-behind-distilling-llm)**
  Accessible intro for non-practitioners.
  *Why for us*: useful as a hook section in a future
  "specialists beat frontier" article.

- **["Need a Small Specialized Language Model? Plan Early!" (arXiv)](https://arxiv.org/pdf/2402.01093)**
  Paper on lifecycle considerations for small specialist LMs.
  *Why for us*: backs the architectural choices we're making for
  Pace planner v6 / v7.

---

## Philosophical / literary takes on LLMs

- **[A Philosophical Introduction to Language Models, Part II (arXiv)](https://arxiv.org/html/2405.03207v1)**
  Academic essay on what's encoded in model weights and what LLMs can
  be said to "know."
  *Why for us*: vocabulary for the "what is the model actually doing"
  framing we keep returning to in mech-interp work.

- **[Reviving the Philosophical Dialogue with LLMs (PhilArchive)](https://philarchive.org/archive/SMIRTP-8)**
  Argument for treating LLM conversations as philosophical exercises.
  *Why for us*: orthogonal reading for thinking about Pace's
  voice-companion shape.

---

## Where this fits

- See `curriculum.md` for the structured 7-session learning track.
- See `docs/learn/ane-research/dossier.md` for our synthesized ANE
  research notes that draw on the Mac/Apple section above.
- See `docs/learn/app-intents-comparison.md` for the App Intents study
  feeding v7 verb taxonomy.
- New entries: add tight one-line `what` + `why-for-us` + link. Don't
  re-explain content that already has an authoritative source.
