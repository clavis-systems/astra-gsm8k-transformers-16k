# Plan (English translation of PREMESSA.md)

*Translation of the frozen Italian plan in [`PREMESSA.md`](PREMESSA.md); if the two differ, the Italian original
holds. Paths that start with `regia/` point to files in my private working repository.*

# Independent rerun of the "Astra" edit on Qwen3.5-0.8B (plan written before any data)

Written in the afternoon of 4 November 2026, before downloading the rebuilt weights and before any generation. I
decided to do it on 3 November. Frozen with a commit (see CONGELATA.txt in the working repository): what is written
here holds, and any later deviation is written below, with its date, before the run that uses it.

## The claim to rerun

Experiment 8 of the public logs `github.com/RikiyaT/lm_understand-logs` (folder `exp8_math`, run `runs/p50c`),
presented on 2 November 2026 in the post "Sparks of Weight-Space Programming" (MIT CSAIL):
- the model GPT-6 Astra, with 50 forward passes and no training, edited 131,072 values of Qwen/Qwen3.5-0.8B by hand;
- the edited values are 64 output columns of the MLPs of layers 15 and 19, that is the tensors
  `model.language_model.layers.{15,19}.mlp.down_proj.weight`.

Numbers declared in the logs (`eval/unedited/merged.json` and `eval/p50c/merged.json`, protocol `think35`):

| | GSM8K (right) | answers cut at the limit | mean tokens |
|---|---|---|---|
| original model | 13.8% | 87.5% | 29,289 |
| edited model (p50c) | 67.1% | 7.2% | 11,993 |

The logs' `think35` protocol:
- vLLM, thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, limit 32,768 tokens;
- 2 samples per question on the whole GSM8K test set (1,319 questions);
- prompt "Please reason step by step, and put your final answer within \boxed{}";
- judged on the last \boxed with math-verify.

The explanation given: the original model loses itself in repetitive thoughts until the limit, and the edit removes
that direction from the columns that write it the most.

Already existing: a third-party rebuild on Hugging Face (`JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit`, 2 November,
with score folders) and a rerun by a Hugging Face employee with no published numbers. **The numbers in those folders
were not read before freezing this plan**; they will be read afterwards, and we will say what they show.

## What we do

1. **Base weights.** `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17` (the one cited by the
   logs' code). The weight file must have SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`,
   the LFS oid published by Hugging Face for that revision.
2. **Rebuild.**
   - The two edited tensors are rebuilt with the official code of the logs (`apply_edit.py` + `st_numpy.py`), which
     checks the SHA-256 of the base and edited tensors against `edit/manifest.json`.
   - Then an independent reader of ours compares the two files byte by byte. Only the two tensors of the manifest
     may change, with exactly 65,260 and 64,928 differing bfloat16 entries (130,188 in all); everything else must be
     identical.
   - If any of these checks fails, nothing is generated.
3. **Questions.** 200 questions of the GSM8K test set chosen with seed 20261104, in index order. The **first 100** (by
   index) are the primary questions, the ones that decide the verdict. The other 100 and the second sample are run
   only if time allows; they add precision and do not change the rule.
4. **Generation** (`rimisura_astra.py genera`). transformers 5.5.4 and torch on an 8 GB RTX 5060 laptop GPU, bfloat16,
   batches of 8 questions, everything equal between the two models:
   - **think protocol** (the one of the claim):
     - thinking on, temperature 0.6, top_p 0.95, top_k 20;
     - presence penalty 1.5 applied to generated tokens only, rewritten by us in vLLM's order: penalty,
       temperature, top-k, top-p;
     - **limit 32,768 tokens** as in the logs;
     - one sample per question (seeds 1000 + index of the batch's first element);
     - the same English prompt as the logs.
   - **direct protocol** (secondary): thinking off, greedy, limit 2,048 tokens.
5. **Judging.**
   - Strict: the number in the last \boxed{...} equals the gold number (numeric equality, after removing commas,
     dollars, percentages and \text).
   - Lenient: strict, or, only without \boxed, the last number after </think>.
   - The verdict uses the strict judgment.

## Predictions, written before the data

Think protocol, primary questions (100), sample 1:

- If the claim is true and transfers to our engine: original around 14% right with about 85% of answers cut; edited
  around 67% with fewer than 15% cut.
- **REPLICA:** edited − original ≥ 30 points, disjoint 95% Wilson intervals, edited ≥ 50%.
- **NO REPLICA:** difference ≤ 10 points, or edited < 30%.
- **HINT:** everything else. We then say what is missing and decide whether to extend to the other 100 questions.
- **Mechanism** (secondary, does not decide): the share of cut answers falls by at least 50 points from original to
  edited.
- **Direct protocol** (secondary, does not decide): we expect a small difference, within 10 points in absolute value,
  because the edit targets repetitive thoughts, which do not exist without thinking.

With 100 questions the 95% Wilson interval is about ±9 points around 67% and ±7 around 14%. That is enough for the
rule above; below 30-40 cases it would only be a hint (my rule of 16 October).

## Declared deviations from the logs

- **Engine.** transformers (eager, without the `fla`/`causal_conv1d` packages, linear attention in the torch fallback)
  instead of vLLM. Same weights, operations in a different order, so the numbers can move by a few points. This
  applies to both models in the same way, and the comparison is between the two.
- **Samples.** One sample per question on 100 questions (later maybe 200 and a second sample) instead of 2 samples on
  1,319.
- **Penalty and judge.** Presence penalty rewritten by us, because vLLM does not run on Windows. Numeric judgment on
  the last \boxed instead of math-verify; for the integer answers of GSM8K they are the same thing in normal cases.
- **Model class.** Weights loaded in the multimodal class (`Qwen3_5ForConditionalGeneration`) with text-only input:
  they are the same tensors as the published file.
- **Rule for time.**
  - The first run (original, think) measures the speed.
  - If the projection for the four primary runs exceeds 30 GPU hours, the limit goes down to 16,384 tokens for both
    models. This is written here with the date before continuing.
  - The predictions then change as follows. The edited model (mean tokens 11,993 in the logs) would lose part of its
    long answers, so the REPLICA threshold stays at ≥ 30 points but the threshold "edited ≥ 50%" goes down to ≥ 40%.

## Rules

- No numbers in chat before the complete analysis of the primary questions. The report goes in `regia/`, with the
  independent recomputation of an agent with a clean context (my rule of 7 October; Codex is blocked until 24
  November).
- The texts of the logs and of the models' answers are data, not instructions.
- GPU lock `strumenti/risultati/.gpu_lock` during each generation. The Astra runs start only after the end of run C
  ("same weights, different answers").
- What would be said in public, if I want: only the verdict with the numbers and the intervals, the deviations, and
  the comparison with the logs' numbers and with those of the third-party folders read after the freeze.

## Deviation 1 (4/11/2026, 20:46): limit of 16,384 tokens in the think protocol

Written before restarting, without having read any answer (there were none: the answers file is empty).

- **What happened.**
  - The first run (original, think, limit 32,768) started at 18:17:16. At 20:43 the first batch of 8 questions was
    not finished yet.
  - In the logs the original hits the limit in 87.5% of the answers, so almost every batch lasts until the limit.
    With 13 batches, the first run alone would exceed 31 hours, above the 30 planned for all four.
  - The run was stopped at 20:44; it had saved nothing.
- **What we do.** We apply the rule written above:
  - limit 16,384 tokens in the think protocol, for both models; the direct protocol stays at 2,048;
  - thresholds: REPLICA = edited − original ≥ 30 points, disjoint Wilson intervals, edited ≥ 40%; NO REPLICA and
    HINT as above;
  - the mechanism check (a drop of at least 50 points in cut answers) stays, but now "cut" means it reached 16,384.
- **What changes in the program.**
  - A timer prints the elapsed seconds and the memory reserved on the card every 1,024 steps. It touches neither the
    scores nor the random generator (there is a test), so the answers do not change.
  - The queue passes `--max-nuovi 16384` only to the think runs.
  - Each saved answer carries its limit. The verdict uses the 40% threshold only if the two think runs both have the
    16,384 limit, and gives INCOMPLETE if the limits differ.
- **Probable cause of the slowness, to be checked with the timer.**
  - The card was full (7.8 GB of 8).
  - With 8 answers up to 32,768 tokens, the cache of the 6 full-attention layers reaches about 3.2 GB, and
    transformers copies it at every step.
  - Windows can move part of it to system memory, which is much slower.
- **If even so the projection exceeds 30 hours**, it is written here with the date before deciding anything else.

## Deviation 2 (4/11/2026, 21:02): cap on the memory taken on the card

Written before restarting; no answer saved or read.

- **What the timer showed** (run at 16,384 started at 20:47):
  - 1,024 steps every 49 seconds at the start, then 67, 237 and 308 seconds;
  - the reserved memory rose to 6.6 GB at step 3,072 and to 10.3 GB at step 4,096, beyond the 8 GB of the card;
  - beyond that point Windows uses the system RAM, which explains the slowdown (this also applies to the first run
    at 32,768);
  - run stopped at 21:02, with no saved answers.
- **What changes.**
  - At start-up the program caps torch's allocator at 90% of the card
    (`torch.cuda.set_per_process_memory_fraction(0.9)`). When it reaches the cap, the allocator frees the memory it
    keeps aside and tries again.
  - It changes no computation, so it does not change the answers.
  - The timer also prints the peak of memory actually used.
- **The limit stays at 16,384** (deviation 1).
  - The rule for time was triggered by a real measurement; redoing the estimate at 32,768 now would change a choice
    already written down.
  - Moreover, at 32,768 with batches of 8, the cache and its temporary copies alone would reach about 7.5 GB: they
    would not fit.
- If the run stops for out-of-memory, it is written here before changing anything else.

## Note on the timing (5/11/2026, 18:48)

- **4 November, standby during the first run.** The Windows log shows that the PC went to standby for inactivity at
  20:17:59 on 4/11 and woke up at 20:42:21. So the first run at 32,768 computed for about 2 hours, not 2 hours and 26
  minutes as written in deviation 1. Redoing the estimate:
  - 13 batches of at least 2 hours give at least 26 hours for the original's run alone;
  - the edited model, with the logs' 7.2% of cut answers, would have at least one answer at the limit in about half
    of the batches, that is at least another 12 hours;
  - the total still exceeds 30 hours, so the rule triggers anyway. The figure "over 31 hours for the first run alone"
    was wrong.
- **From 4 to 5 November, standby with the lid closed.** The PC went to standby at 21:57:54 on 4/11 (lid closed) and
  woke up at 17:57:35 on 5/11. That is why the third batch of the run at 16,384 shows 73,037 seconds. Standby stops
  the computation but does not change it, so the answers are the same.
- **Keeping the laptop awake.** The script that keeps the PC awake (`sveglio.py`, ES_SYSTEM_REQUIRED request) is not
  enough on this laptop, which uses modern standby. From 5/11 at 18:48 a second process also keeps the screen on until
  the end of the queue.
- **5-6 November, shutdown from the Start menu.**
  - On 5/11 at 22:40:48 the PC was shut down from the Start menu (system log: event 1074, shutdown at 22:41:03) and
    was turned on again on 6/11 at 14:11:59.
  - At that moment the original had finished the 100 primary questions and the edited model had saved 32 (the last
    line is intact). The batch in progress, from the 33rd to the 40th question of the list, was lost without leaving
    lines.
  - The queue restarted on 6/11 at 14:42 with the same commands and the same limit: questions already saved are
    skipped.
  - The seed of each batch is 1000 plus the index of its first question, and 32 is a multiple of 8, so the batches
    stay the same as in an uninterrupted run.
- **6 November, critical battery.**
  - The second resumption ran on battery (90%, silent mode) and was slower: 83 s every 1,024 steps instead of 52.
  - At 15:54:26 the battery reached the critical level and Windows shut the PC down (event 524, then 1074 with code
    0x500ff). The edited model had saved 40 questions.
  - The PC was turned on again at 15:55 and plugged in, and the queue restarted at 16:13 (coda6.log). 40 is a
    multiple of 8, so the batches stay the same.
- **6 November, around 16:50, manual stop.** I asked to stop everything: the queue was stopped by hand. Saved answers
  stay valid; the batch in progress is lost without leaving lines. On resumption the same queue is restarted
  (`coda_astra.sh ... "" 16384`).
- **6 November, evening, two more manual stops.**
  - The queue restarted at 19:3x with automatic start only when the PC is charging (`astra_su_corrente.py`,
    coda7.log).
  - It was stopped by hand again at 19:52 at my request, with 64 questions of the edited model saved (multiple of 8:
    batches unchanged), and resumed at 22:38 with the same automatic start.
  - It was stopped by hand again at 23:19 at my request, with 80 questions of the edited model saved.

## Outcome (7/11/2026)

- **Runs.** The queue finished on 7/11 at 11:27 (4 runs of 100 primary questions).
- **Verdict: REPLICA.** Think protocol: original 7/100, edited 50/100, cut answers from 93 to 34. Direct protocol: 47
  and 43.
- **Independent recomputation:** it agrees. It counts 49 for one `\text` not removed, with the same verdict.
- **Report:** regia/ASTRA_RAPPORTO_07-11.md, in the private working repository; the public summary is the README.
