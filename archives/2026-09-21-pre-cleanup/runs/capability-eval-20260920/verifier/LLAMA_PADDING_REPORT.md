# Llama generation padding fix: independent verification

Status: PASS. Before execution, GPU7 had no capability-evaluation process; its existing unrelated 626 MiB Python process was left untouched. Verification has finished and GPU7 was released.

The adapter now assigns `tok.pad_token_id = tok.eos_token_id` before HFLM construction for Llama. The runtime IDs are pad/EOS 128009, UNK 128256, and 128256 embedding rows. HFLM's `configure_pad_token` otherwise prefers UNK when no pad token exists, producing an invalid index for this tokenizer/model combination. Setting an existing valid EOS as padding fixes that boundary without resizing embeddings or modifying weights/scales.

`check_llama_padding.py` loaded the final Llama FIRON package on GPU7, used actual `HFLM.tok_batch_encode` on prompts of 5 and 15 tokens, and called actual `HFLM._model_generate` for a batch of two, with two new tokens. The short prompt received ten left-padding IDs 128009 with attention-mask zeros. All IDs were within embedding bounds. Generation succeeded; all 112 quantizers executed prefill and decode; every quantizer state tensor was unchanged. Raw IDs, masks, counts and result are in `llama_padding.json`.

Completed likelihood-scoring tasks are unaffected: harness `huggingface.py` causal `_loglikelihood_tokens` calls `pad_and_concat(..., padding_side="right")`; `models/utils.py` implements that padding with `torch.zeros`, independently of tokenizer.pad_token_id. Continuation logits are selected using the original input and continuation lengths. The fix changes neither prompt tokenization nor scoring padding, so zero-shot and MMLU results do not need rerunning because of this issue.
