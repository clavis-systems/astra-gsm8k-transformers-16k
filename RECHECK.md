# Independent recheck of the Astra rerun

*English translation of the recheck report, written in Italian on 7 November 2026 by a separate AI agent session (Claude)
with a clean context. Its scripts are in `recheck/code/`; only their file paths were changed for this repository.*

Done from PREMESSA.md and the raw answer files only. I did not read the analysis script, the tests, the queue,
analisi.json or the final part of the queue log.

My code: calc.py (main computation), mano.py (cases by hand, seed 20261107), var2.py (variant), c52.py.

Basis: 100 primary questions, sample 1, for each run. Intervals: Wilson 95%.

## 1. Completeness

- The four runs have exactly the 100 primary questions (the first 100 indices of domande.jsonl), each once.
- Limit: 16,384 for both think runs, 2,048 for both direct runs. No mixed limits, so the 40% threshold applies.
- `tagliata` (cut) matches `n_token >= tetto` in all 400 lines (0 inconsistencies).

## 2. Numbers per run (strict judgment, base parser)

| run | strict right | Wilson 95% | lenient | cut | mean tokens | without \boxed |
|---|---|---|---|---|---|---|
| original, think | 7 | 3.4 – 13.7 | 7 | 93 | 15,564 | 61 |
| edited, think | 49 | 39.4 – 58.7 | 51 | 34 | 11,818 | 31 |
| original, direct | 47 | 37.5 – 56.7 | 47 | 11 | 612 | 11 |
| edited, direct | 43 | 33.7 – 52.8 | 43 | 9 | 615 | 9 |

The "without \boxed" count of the base parser also includes answers whose last \boxed is empty (see below). The real
number of answers with no \boxed at all is lower in the original's think run.

## 3. Paired

- Think: both 7, only edited 42, only original 0, neither 51.
- Direct: both 36, only edited 7, only original 11, neither 46.

## 4. Verdict (rules applied literally, with deviations 1 and 2)

- **Think difference:** 49 − 7 = +42 points (threshold: at least 30). Passes.
- **Intervals:** [3.4 – 13.7] and [39.4 – 58.7], disjoint. Passes.
- **Edited threshold:** 49% (threshold with the 16,384 limit: at least 40%). Passes. With the original 50% threshold
  it would not pass, but the limits are 16,384 for both, so 40% applies.
- **Verdict: REPLICA.**
- **Mechanism** (secondary): cut answers 93 → 34, a drop of 59 points (threshold 50). Present.
- **Direct** (secondary): 43 − 47 = −4 points, within 10. The prediction is confirmed.
- **Against the claim:**
  - The original scores 7% (logs: 13.8%) with 93% cut (logs: 87.5%).
  - The edited model scores 49% (logs: 67.1%) with 34% cut (logs: 7.2%).
  - The limit here is halved, so part of the distance is expected. The sign and the mechanism are those of the logs;
    the size is not.

## 5. Robustness

### Interpretations

- **Base parser.**
  - Last `\boxed{...}` with balanced braces, so nested ones are fine.
  - Inside it I remove `$`, `\$`, `%`, `\%`, commas, `\,`, `\!`, `\left`, `\right`, and `\text{..}` (keeping its
    content).
  - Then I require an integer or decimal number, or `\frac{a}{b}` / `\dfrac`.
  - Exact equality with Fraction, so "70,000" and "70000.00" count as 70000.
- **Units.** A `\boxed{12 dollars}` or `\boxed{188 \text{ mm}}` with unit text does not read as a number, so it is
  wrong in the base parser. The "lenparse" variant takes the last number inside the box.
- **Empty boxes.** Many answers, above all those cut during thinking, contain an empty `\boxed{}` quoted from the
  prompt ("put your final answer within \boxed{}").
  - Literally, "the last \boxed" is the empty one, so the answer is wrong, and it counts as "with \boxed" but not
    readable. In the base parser it is so.
  - The "nonvuoto" (non-empty) variant ignores empty boxes.
- **`\boxed` inside the thinking of a cut answer.**
  - In the base parser it counts (literal reading). No cut answer turns out right in any run (checked).
  - The "dopothink" (after-think) variant counts only boxes after `</think>`, and for cut answers without `</think>`
    no box. It gives the same scores for the right answers (7, 49, 47, 43).
- **Lenient.** If there is no \boxed at all and there is `</think>`, the last number after `</think>`, with commas
  and symbols removed. Without `</think>`, no fallback.

### Strict vs lenient

Only 2 answers change outcome, both in the edited think run (original think and both direct runs: 0).
- i=138: "Final Answer: 105." without \boxed. Right; the fallback is legitimate.
- i=52: answer cut at 16,384 while it was writing the final answer, after `</think>`. Its last number (15) matches
  the gold value, but the outcome is accidental. A doubtful fallback.

### Parser variants (think, original vs edited, right out of 100)

| variant | original | edited | difference | disjoint | edited ≥ 40% |
|---|---|---|---|---|---|
| base parser | 7 | 49 | +42 | yes | yes |
| lenparse (last number in the box) | 7 | 50 | +43 | yes | yes |
| dopothink | 7 | 49 | +42 | yes | yes |
| nonvuoto + lenparse | 9 | 50 | +41 | yes (4.8–16.2 vs 40.4–59.6) | yes |

- No variant changes the verdict: REPLICA.
- Direct: unchanged under any variant (47 / 43).
- The edited model never goes below 49.
- The nonvuoto variant moves two right answers into the original's count: i=220 and 452, a correct box inside the
  thinking of a cut answer, after an empty box. It is the only point where the choice moves the original.

### Check by hand

Edited think run, seed `random.seed(20261107)`, 5 right and 5 wrong among the primary questions.
- **Right:** the judgment is correct in 5 out of 5, and the reasoning is consistent with the result.
  - i=491: an empty box first, then last \boxed{16}, gold 16;
  - i=594: 42, gold 42;
  - i=412: 5, gold 5;
  - i=451: an empty box first, then \boxed{11050}, gold 11050, with a comma in the text;
  - i=24: 26, gold 26, 15,641 tokens.
- **Wrong:** the judgment is correct in 5 out of 5.
  - i=124: cut, no box, gold 10;
  - i=540: cut, gold 39;
  - i=417: cut, gold 3140; it was already after `</think>` and writing out the steps;
  - i=236: box 25, gold 31. Really wrong: "Final Answer: \boxed{25}" after reasoning that says it does not converge;
  - i=192: cut, gold 91.
- **Also watched:** all cases where the last box does not read as a number.
  - Empty boxes, above all in the original's think run.
  - `188 \text{ mm}` (i=537, gold 188, edited think): it would be right with a parser lenient on units (see
    lenparse).
  - `33 \frac{1}{3}` (i=64 direct, gold 300): wrong anyway.
  - `Steve` (i=150 direct, gold 4): wrong.

*Note added by the repository author: the plan says to remove `\text` before comparing. The main analysis drops
`\text{...}` together with its content, so i=537 counts as right there (50). The base parser here kept the word "mm",
so it did not read as a number (49). The verdict is the same.*

## 6. Weights

verifica_pesi.json:
- **Base:** revision 2fc06364715b967f1860aea9cf38778875588b17 and SHA-256 of the base file 04b1c301...4696, as in
  the plan.
- **Changed tensors:** exactly 2 tensors change, layers.15 and layers.19 `mlp.down_proj.weight`, with 65,260 +
  64,928 = 130,188 differing bfloat16 entries. Everything else is identical (488 tensors compared; only-base and
  only-edited empty).
- **Checks:**
  - All four checks are true: same names, only the manifest's tensors change, differing entries as in the manifest,
    hashes as in the manifest.
  - Changed share: 0.0149% of the file.
- **What I could not do:** the file tells me the checks were done. I cannot redo the byte-by-byte comparison (I do not
  have the weights). Point 2 of the plan is satisfied.

## Limits to declare

- 100 questions, one sample, limit 16,384 (not 32,768 as in the logs). Strong difference, but the absolute numbers
  are not those of the logs.
- 93 of the original's 100 answers are cut. So the comparison mostly measures "the original does not finish and the
  edited model finishes more often"; it is not a measure of reasoning ability.
