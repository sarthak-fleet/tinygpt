# Dataset inventory

Updated by agents running `docs/prds/dataset-decode-verify.md`.
The decoded JSONLs live under `~/.cache/tinygpt/datasets/` and are not
checked into the repo.

| Dataset | Local JSONL | Rows | Columns | Intended use | Notes |
|---|---:|---:|---|---|---|
| `princeton-nlp/SWE-bench_Verified` | `swe-bench-verified.jsonl` | 500 | `repo`, `instance_id`, `base_commit`, `patch`, `test_patch`, `problem_statement`, `hints_text`, `created_at`, `version`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `environment_setup_commit`, `difficulty` | Code specialist eval / training | Decoded from one cached parquet shard; `head -3` JSON validation passed. |
| `iamtarun/python_code_instructions_18k_alpaca` | `python-code-instr.jsonl` | 18612 | `instruction`, `input`, `output`, `prompt` | Code specialist SFT | Decoded from one cached parquet shard; `head -3` JSON validation passed. |
| `Locutusque/function-calling-chatml` | `function-calling-chatml.jsonl` | 112960 | `system_message`, `function_description`, `conversations` | A1 tool-caller SFT | PRD named `NousResearch/function-calling-chatml`, but the landed cache is `Locutusque/function-calling-chatml`; decoded from one cached parquet shard; `head -3` JSON validation passed. |
| `bigcode/the-stack-smol` | _not decoded_ | 0 | _none_ | Code specialist pretrain | Cache directory exists, but contains no files/parquet shards. |
| `NousResearch/function-calling-chatml` | _not decoded_ | 0 | _none_ | A1 tool-caller SFT | Exact PRD target was not pulled; see decoded `Locutusque/function-calling-chatml` row above. |
| `microsoft/ms-marco` | _not decoded_ | 0 | _none_ | B25 ScaleDown pretrain | Cache directory missing. |
| `google-research-datasets/natural_questions` | _not decoded_ | 0 | _none_ | B25 ScaleDown pretrain | Cache directory missing. |
