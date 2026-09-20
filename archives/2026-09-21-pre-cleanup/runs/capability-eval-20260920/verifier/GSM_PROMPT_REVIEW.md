# GSM8K prompt-structure review

Conclusion: retain the already declared protocol. The inspected structure is an explicitly supported harness single-user-turn chat adaptation, not an implementation error requiring a switch to multiturn. This conclusion uses prompt construction and source evidence, not benchmark scores. No generation, parameter changes or alternative-prompt experiment was performed.

Local official harness 0.4.8 evidence:

- `lm_eval/__main__.py:185-188`: `--fewshot_as_multiturn` is an optional flag with default False.
- `lm_eval/api/task.py:1030-1055`: the fewshot-context API documents the choice as multiturn conversation versus single user turn.
- `lm_eval/api/samplers.py:171-179`: the False branch deliberately places `get_context(...)` (all demonstrations) in one user message.
- `lm_eval/api/task.py:1017-1023`: the target question is deliberately appended to that last user message when multiturn is False.
- `lm_eval/tasks/gsm8k/gsm8k-cot.yaml`: the task specifies its eight fixed Q/A CoT demonstrations, target question template and extraction filters, but does not mandate chat-template or multiturn use. Pure evaluator defaults have chat formatting disabled, so the present run should be described as an adaptation, not unchanged official default completion prompting.

The first raw FIRON Llama prompt was decoded from `llama/firon/gsm8k_cot/generation_batches-001.jsonl`, reading only prompt IDs. It contains nine `Q:` and nine `A:` markers, eight demonstration answer endings, and the target problem last. The assistant-generation header follows the target's final `A:`. No missing target, shifted question, or target-answer leakage was found.

The paper should state: **GSM8K 8-shot CoT, model chat template, demonstrations and target in a single user turn, strict extraction, greedy, 256-token cap**. The formulation differs from a pure completion prompt and from a multiturn demonstration dialogue, so comparisons to reported external GSM8K numbers need that qualification. A model answering examples instead of the final question is a prompt-format failure under this selected protocol; it does not by itself prove the adapter violates its stated task semantics. It also limits how strongly very low accuracy can be attributed to lost reasoning ability alone. Switching to multiturn after seeing outputs would define another protocol and should not silently replace these results.

Additional provenance detail: the model-supplied Llama template inserts a system date (`20 Sep 2026` in the inspected prompt). Final within-model prompt matching should include that date if runs cross calendar days.
