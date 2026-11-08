# The Astra weight edit on Qwen3.5-0.8B: a GSM8K rerun with transformers and a 16K budget

Emanuele Rizzan, independent researcher (Italy). November 2026.

The post "Sparks of Weight-Space Programming" (MIT CSAIL, 2 November 2026) reports that GPT-6 Astra edited about
130,000 weights of Qwen3.5-0.8B by hand, with no training, and that with thinking on the model's GSM8K accuracy went
from 13.8% to 67.1%. The explanation given: the original model loops in its thinking until the token limit, and the
edit removes that direction.

I reran the comparison on a different stack: Hugging Face transformers instead of vLLM, on an 8 GB laptop GPU, with a
16,384-token limit instead of 32,768. The verdict rule was written down before any generation.

## Result (100 GSM8K test problems, one sample each)

| | right (strict) | 95% Wilson interval | hit the token limit | mean tokens |
|---|---|---|---|---|
| original, thinking on | **7** | 3.4–13.7 | 93 | 15,564 |
| edited, thinking on | **50** | 40.4–59.6 | 34 | 11,818 |
| original, thinking off (greedy, 2,048 tokens) | 47 | 37.5–56.7 | 11 | 612 |
| edited, thinking off (greedy, 2,048 tokens) | 43 | 33.7–52.8 | 9 | 615 |

Paired by problem, thinking on: 43 problems are right only with the edit, 0 only without it, 7 with both and 50 with
neither.

**Verdict under the pre-written rule: REPLICA.** The edited model is 43 points higher, the intervals do not overlap,
and it is above the 40% threshold set for the 16K budget. The share of answers that hit the limit drops by 59
points, and the pre-written mechanism check asked for at least 50. With thinking off the difference is −4 points; the
prediction was "within 10".

## Next to the other numbers

| | original, thinking on | edited, thinking on | edited, hit the limit | thinking off, original / edited |
|---|---|---|---|---|
| authors' logs (vLLM, 32,768 tokens, 1,319 × 2) | 13.8% | 67.1% | 7.2% | — |
| [JonathanColetti's rerun](https://huggingface.co/JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit) (vLLM, 32,768, 1,319 × 2) | 13.0% | 67.6% | 7% | 57.1% / 55.7% (their settings) |
| this rerun (transformers, 16,384, 100 × 1) | 7% | 50% | 34% | 47% / 43% (greedy) |

What this adds is small:
- the direction of the effect and the drop in truncation hold outside vLLM, at n=100;
- the size is smaller than in the logs. The lower budget is the likely reason: the edited model thinks long (11,818
  tokens on average), and at 16K a third of its answers still run out. But the engine and the sample also differ, so I
  cannot separate the causes;
- with thinking off the edit does not help, as in JonathanColetti's rerun.

It is a further confirmation, not the first. JonathanColetti had already rerun the full test set with the authors'
engine, and other checks are collected in the issues of the
[authors' log repository](https://github.com/RikiyaT/lm_understand-logs/issues).

## Another researcher re-ran this protocol

On 8 November 2026 [HedonismJesse](https://github.com/HedonismJesse) posted a re-run of this protocol on five versions
of the weights, with the 100 primary questions in `data/domande.jsonl`, the same batch seeds, and predictions written
down before the results: [the comment with the results](https://github.com/RikiyaT/lm_understand-logs/issues/8#issuecomment-6719824287).

| | their run | this repository |
|---|---|---|
| Thinking on, original / edited, right of 100 | 12 / 54 | 7 / 50 (49 with a judge that keeps units) |
| Thinking on, right only with the edit / only without it | 42 / 0 | 43 / 0 (42 / 0 with that judge) |
| Thinking off, original / edited, right of 100 | 46 / 45 | 47 / 43 |
| Thinking off, right only with the edit / only without it | 10 / 11 | 7 / 11 |

These are their numbers: I have not recomputed them. Each run has one sample per problem, so the thinking-off
differences are compatible with no change.

## How it was done

1. **Weights.**
   - Base: `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`; the weight file's SHA-256 is
     checked against the LFS oid.
   - The two edited tensors were rebuilt with the authors' own `apply_edit.py`, which checks the tensor hashes
     against their manifest.
   - A separate reader then compared the two files: exactly 130,188 bf16 entries differ (65,260 + 64,928) in
     `layers.15` and `layers.19` `mlp.down_proj.weight`, and every other tensor is identical
     (`data/verifica_pesi.json`).
2. **Questions.** 200 GSM8K test problems chosen with seed 20261104. The first 100 by index are the primary ones and
   decide the verdict. Only the primary ones were run.
3. **Generation.** transformers 5.5.4, bf16, batches of 8, the authors' prompt. Two protocols:
   - thinking on: temperature 0.6, top_p 0.95, top_k 20 and presence penalty 1.5 on generated tokens only. vLLM does
     not run on Windows, so I reimplemented the penalty in vLLM's order;
   - thinking off: greedy, 2,048 tokens.

   Each batch's seed is 1000 plus the GSM8K index `i` of its first question, so a run that was interrupted and resumed
   gives the same answers.
4. **Judging.**
   - Strict: the number in the last `\boxed{...}` equals the gold answer, after removing commas, dollar and percent
     signs and `\text{...}`.
   - Lenient: strict, or, only when there is no `\boxed`, the last number after `</think>`. When there is no
     `</think>` either, the code takes the last number in the whole text. That looser fallback gives `acc_lenta`
     13% and 55% in `data/analisi.json`; the recheck's lenient rule has no fallback without `</think>` and gives 7%
     and 51%.
   - The verdict uses strict only.

## Deviations, written down before the runs that used them

The plan is in [`PREMESSA.md`](PREMESSA.md), the frozen original in Italian; [`PLAN.md`](PLAN.md) is an English
translation.
- **Deviation 1: the token limit.** The first run at 32,768 tokens was too slow for this laptop: the projected time
  was over 30 hours, and the plan said to lower the limit in that case. The limit went down to 16,384, and the
  threshold for the edited model from 50% to 40%. Both were written down before the run that used them, with no
  answers saved or read.
- **Deviation 2: the GPU memory cap.** Torch's allocator was capped at 90% of the card. This does not change any
  computation.
- **The interruptions.** The runs were stopped several times (standby, shutdown, low battery, manual stops). This is
  logged in `PREMESSA.md`; resumption skips saved questions and keeps the same batches.

## Independent recheck

A separate AI agent session recomputed everything from the plan and the raw answers only, with its own code: see
[`RECHECK.md`](RECHECK.md) and `recheck/code/`. It did not read my analysis script.
- **Its verdict:** REPLICA.
- **The one disagreement:** problem 537, written `\boxed{188 \text{ mm}}` with gold 188.
  - The plan says to remove `\text`. The main analysis drops `\text{...}` together with its content, so the answer
    counts as right (50).
  - The recheck's base parser removed the command but kept the word "mm", so it did not read as a number (49).
  - The verdict is the same either way, and no parser variant the recheck tried changes it.
- **Checked by hand:** 10 answers, judged correctly in 10 cases out of 10.

## Limits

- 100 problems, one sample each: the 95% intervals are up to about ±10 points.
- The token limit is 16,384 instead of 32,768, so the absolute numbers are not the authors'.
- The engine is different and the presence penalty is reimplemented.
- In thinking mode the original model almost never finishes (93% truncated). So the comparison mostly measures
  "finishes within the limit", not reasoning ability.
- The plan was frozen in my private repository on 4 November 2026. This public repository publishes it afterwards, so
  for the order of events you have my word and the dated plan, not a public commit from before the runs.
- JonathanColetti's numbers were already public when I froze the plan; I did not read them until after the analysis.

## Files

The file names and the code comments are in Italian.

| path | what |
|---|---|
| `PREMESSA.md` / `PLAN.md` | the frozen plan (Italian original) and its English translation |
| `RECHECK.md` | the independent recheck (English) |
| `data/domande.jsonl` | the 200 chosen questions (`i` = GSM8K test index, `domanda` = question, `oro` = gold answer) |
| `data/risposte/*.jsonl` | the 400 answers. `base` = original, `modificato` = edited; `pensa` = thinking on, `diretto` = thinking off; `c1` = sample 1 |
| `data/verifica_pesi.json` | the weight checks (`diversi` = tensors that differ, `voci_diverse` = differing entries) |
| `data/analisi.json` | the analysis output, verdict included: `corse` (runs), `acc_stretta`/`ic95_stretta` (strict accuracy and interval), `acc_lenta` (lenient), `quota_tagliate` (hit the limit), `token_medi` (mean tokens), `senza_boxed` (no readable `\boxed`), `primarie`/`tutte` (primary/all questions), `appaiato` (paired: `entrambi_giusti` both right, `solo_modificato` only edited, `solo_base` only original, `nessuno` neither), `verdetto` (verdict) |
| `code/rimisura_astra.py` | rebuild and checks (`prepara`), generation (`genera`), analysis (`analizza`) |
| `code/coda_astra.sh` | the queue of the four runs; `code/test_rimisura_astra.py`: tests |
| `recheck/code/` | the recheck's own scripts |

Each answer line has `i`, `campione` (sample), `risposta` (the generated text), `n_token`, `tagliata` (hit the limit),
`tetto` (the limit) and timing fields.

## Recompute

```bash
cd code
python rimisura_astra.py analizza --out ../data    # rewrites data/analisi.json; it should not change
python test_rimisura_astra.py
cd ../recheck/code
python calc.py                                     # the recheck's numbers
```

On Windows the analysis may rewrite `data/analisi.json` with different line endings, so git shows a diff; the numbers
do not change.

To regenerate the answers you need the base weights, the authors' `exp8_math` edit files and code, and a GPU. See
`python rimisura_astra.py prepara --help` and `genera --help`. Some default paths in `code/` (the GSM8K file, a GPU
lock file) assume my private folder layout: pass your own `--base` and `--gsm8k`. Paths in `PREMESSA.md` (`regia/`,
`strumenti/`, CONGELATA.txt) also refer to my private working repository.

## Credits and links

- Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL): the post, the edit and the logs,
  https://github.com/RikiyaT/lm_understand-logs. Their edit and logs are CC BY 4.0 and their scripts MIT. The answers
  in `data/risposte/` come from a model carrying that edit; this repository does not redistribute the edit files or
  their code.
- JonathanColetti's rerun: https://huggingface.co/JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit.
- GSM8K (Cobbe et al., 2021), MIT license.
- Qwen3.5-0.8B (Qwen team), Apache-2.0.

## Who did what

I chose to rerun this, set the rules (plan written before the data, an independent recheck, report whatever comes
out) and made the decisions. The code, the runs and the analysis were done by AI coding agents (Claude) under my
direction. The recheck was done by a separate agent session that did not see my analysis code.

## License

Copyright 2026 Emanuele Rizzan. Code and text: Apache-2.0 (see `LICENSE`). The questions come from GSM8K (MIT). The
answers are outputs of Qwen3.5-0.8B, original and edited.
