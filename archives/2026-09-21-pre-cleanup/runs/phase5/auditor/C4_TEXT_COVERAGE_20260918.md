# Fixed C4 source-text coverage by tokenizer

2026-09-18. **Coverage resolved for both models; the legacy Llama raw offset mapping has a documented limitation.** This fills handoff §6.3 source-text coverage fields using read-only CPU analysis. No GPU, PPL rerun, document sampling, new hash, application change or modification to existing token/metadata artifacts. Full evidence: `C4_TEXT_COVERAGE_20260918.json`; per-model details: `c4_text_coverage_llama_20260918.json` and `c4_text_coverage_qwen_20260918.json`.

Both tokenizers use the same ordered **4,480** documents from `runs/phase2/c4-acceptance-c-20260912.FJXr6U/data/documents.jsonl`, joined with `\n\n`, without added special tokens/BOS/EOS. Document text totals **10,089,434 characters**, separators **8,958**, and joined text **10,098,392**. Here “character” means a Python Unicode code point, not a UTF-8 byte or grapheme cluster; positions are zero-based half-open spans. Document ordinal is explicitly one-based below; source `row_index` retains its original zero-based dataset index.

| Coverage field | Llama-3.2-1B-Instruct | Qwen3-1.7B |
|---|---:|---:|
| Full joined-text token count | 2,150,947 | 2,191,766 |
| Retained input tokens | 2,097,152 | 2,097,152 |
| Discarded suffix tokens | 53,795 | 94,614 |
| Complete source documents | 4,382 | 4,325 |
| Last touched document ordinal | 4,383 | 4,326 |
| Last touched document source row_index | 171371 | 230792 |
| Last document covered / total characters | 996 / 4,569 | 40,167 / 67,339 |
| Covered source-document characters, excluding separators | 9,832,081 | 9,650,912 |
| Covered separator characters | 8,764 | 8,650 |
| Covered joined-text prefix `[0, end)` | `[0, 9840845)` | `[0, 9659562)` |
| Effective prediction targets | 2,096,128 | 2,096,128 |
| Full windows × length; partial tail | 1,024 × 2,048; none | 1,024 × 2,048; none |

Llama's last document occupies global source span `[9839849, 9844418)`; Qwen's occupies `[9619395, 9686734)`. Llama covers **181,283 more joined-text code points** (181,169 more document code points), despite equal retained token counts. The shared document list therefore does not imply equal retained text. These are input-prefix coverage counts; 1,024 window-initial tokens are not prediction targets, and no claim maps the loss targets one-to-one to source characters. Absolute per-token PPL is not a cross-tokenizer ranking.

## Formal tokenizer identity and correspondence

Llama used `/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python`, Transformers **4.44.2**, tokenizers **0.19.1**, `LlamaTokenizerFast`, model path `cache/models/llama-3.2-1b-instruct`. Qwen used `runs/phase5/env/bin/python`, Transformers **4.51.3**, tokenizers **0.21.4**, `Qwen2TokenizerFast` selected through `AutoTokenizer`, model path `cache/models/qwen3-1.7b`. In each formal environment, tokenization with offsets produced IDs exactly equal to the existing cached 2,097,152-token prefix. This comparison specifically associates the coverage analysis with the evaluated cache; it did not create a new token dataset.

Llama cache: `runs/phase2/c4-acceptance-c-20260912.FJXr6U/data/input_tokens.pt`. Qwen cache: `runs/phase5/qwen3-1p7b/c4-data/input_tokens.pt`. The observed full token counts also equal the existing metadata counts.

## Offset limitations and boundary verification

**Qwen:** retained offset spans merge into exactly `[0, 9659562)`. The last retained token is ` getting`, span `[9659554, 9659562)`; the first discarded token is ` tips`, span `[9659562, 9659567)`. The cut lies inside document 4,326, not in the two-newline separator, and does not split a Unicode character. There are 343 adjacent overlap pairs (173 identical pairs) elsewhere in the retained token offsets and zero zero-length spans. Interior overlapping byte-token spans are counted once; they do not indicate a partial character at this cut.

**Llama:** the formal legacy class returns **1,839,926 zero-length spans** within the retained prefix, including the terminal tokens. The final retained ` fishing` token has raw offset `[9840837, 9840837)`, while the first discarded ` or` has `[9840845, 9840845)`. Consequently neither the raw union nor its last nonempty span establishes text coverage. In particular, the raw nonempty-span maximum of 9,840,815 is **not** the correct prefix endpoint; using it would undercount the last document by 30 characters. Raw diagnostics remain in the evidence JSON, explicitly marked unreliable for coverage. No source/tokenizer behavior was changed to repair these offsets.

To resolve this specific limitation, the **same formal Llama tokenizer decoded the existing cached prefix** with `skip_special_tokens=False` and `clean_up_tokenization_spaces=False`. The resulting complete string exactly equals original joined text `[0, 9840845)` (9,891,960 UTF-8 bytes). It ends in the ASCII word `fishing`; the next source text begins ` or`. This exact source-prefix correspondence, together with the terminal ASCII token pieces, establishes the reported code-point boundary without relying on the zero-width offsets. It does not cut a Unicode character or a separator. The raw retained offsets also contain 395 adjacent overlap pairs (226 identical pairs), but those raw statistics are diagnostics, not a character-coverage measure.

The raw-offset inspection helper `c4_text_coverage.py` now labels zero-width mappings unresolved and writes only raw diagnostic outputs, so it cannot overwrite the resolved final coverage report/JSON on a later run. No second full tokenization was needed for the Llama fallback; it read and decoded the existing prefix only. If a future boundary has retained/discarded Unicode span overlap or incomplete byte decoding, its span and partial-character uncertainty must be reported instead of claiming complete-character coverage.
