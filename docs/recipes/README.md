# TinyGPT Recipes

Copy-paste workflows for using TinyGPT specialists outside the core CLI.

| Recipe | Use it when |
|---|---|
| [Function-calling distillation](distillation-fc.md) | You want to distill a small tool-calling specialist and score it with BFCL. |
| [ScaleDown specialist](b25-scaledown.md) | You want an extractive context-compression specialist. |
| [smolagents specialist](cookbook-smolagents.md) | You want a TinyGPT model behind a Hugging Face smolagents tool-calling agent. |
| [Pydantic AI specialist](cookbook-pydantic-ai.md) | You want structured outputs from a TinyGPT-backed Pydantic AI agent. |
| [Personal code specialist](cookbook-personal-code-specialist.md) | You want a per-repo TinyGPT model wired into Continue.dev or Aider. |
| [Character specialist](cookbook-character-specialist.md) | You want a free local NPC, persona, or brand-voice specialist recipe. |
| [Eval gate (CI / pre-commit)](eval-gate.md) | You want `tinygpt eval-gate` to fail a merge when a specialist regresses, on your own Mac runner. |
