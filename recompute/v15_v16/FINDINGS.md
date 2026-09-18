# Sections 2.7 (v15, v15b) and 2.8 (v16): claim-by-claim recount

Script: `recompute_v15_v16.py` (written from scratch; does not import the project's analysis code).
Output: `recompute_v15_v16.json`. Basis **RAW** = recomputed from `samples.json` rows, the locked request plans
and per-call checkpoints; **DERIVED** = read from a saved report or XML; the project's own audit code was
re-run separately (see `outputs/01_official_audits_summary.json` and `outputs/02_audit_comparison.json`).

**No CONFLICT found in sections 2.7-2.8.** Two design facts that the handoff states only implicitly are
spelled out at the end, because they matter for the next design.

## 2.7 v15 / v15b

| Claim (handoff) | Recomputed | Basis | Status |
|---|---|---|---|
| 24 instances, 9 families, 12 truth + 12 validity | 24 tasks; 9 families; truth 12 / validity 12 | RAW (plan) | MATCH |
| Target direction balanced | target YES 12 / NO 12 | RAW (plan) | MATCH |
| Task given in the first history user message and restated at the end | 480/480 study requests: first history message starts with the task body; final message contains the task (or the updated task in the update arm) | RAW (plan requests) | MATCH |
| Three history user turns + three `Noted.` | 480/480 | RAW | MATCH |
| Calibration 96/96, all 24 items qualified | 96/96 correct; 24/24 items with 4/4 correct | RAW | MATCH |
| neutral 96/96; last-only 96/96; repetition 95/96; preference 96/96; update 80/96 | identical | RAW | MATCH |
| Preference - repetition = -1.04 pp, 9-family bootstrap [-3.75, 0] pp | -1.0417 pp; [-3.75, 0.00] pp (seed 2026091602, 10,000 family resamples); 34.4 % of resamples are exactly 0 | RAW (re-implemented) | MATCH |
| The difference comes from a single repetition-arm error | 1 wrong: repetition, `denying_antecedent`, item `v15_denying_antecedent_0` | RAW | MATCH |
| 96 preference/repetition pairs have identical input-token counts | 96/96 pairs with difference 0 | RAW (metadata) | MATCH |
| 16 update errors, all validity, modus ponens and disjunctive syllogism, 4 lexical instances | 16 = MP 8 + DS 8; 4 items x 4 | RAW | MATCH |
| v15b standalone 96/96; history - standalone +16.67 pp, [0, 44.44] | 96/96 (truth 48/48, validity 48/48); +16.67 pp; [0.00, 44.44] pp | RAW (re-implemented) | MATCH |
| 672 new calls = 96 + 480 + 96, all EOS, zero unknown/truncation | 672 rows, 672 unique seeds, 672 EOS, 0 unknown, 0 backend errors; 672 completed checkpoints | RAW | MATCH |
| Zero-call resumes pass | re-run through the Windows adapter: all three resumes 0 new calls, raw hashes unchanged | Official audit re-run | MATCH |
| 276 tests in each environment | saved `pytest_original_v15b.xml` and `pytest_mlx_v15b.xml`: 276 tests, 0 failures/errors/skips | DERIVED (saved XML) | MATCH |

## 2.8 v16

| Claim (handoff) | Recomputed | Basis | Status |
|---|---|---|---|
| 4 families x 2 lexical instances | families: disjunctive_syllogism, forward_chain, modus_ponens, modus_tollens; 8 pairs | RAW | MATCH |
| 64 calibration all correct | 64/64 | RAW | MATCH |
| 512 study, 428 correct, 84 wrong, 576 total, all EOS, no unknown/truncation/backend error | 512: 428 / 84 / 0; 576 unique seeds, 576 EOS | RAW | MATCH |
| Cell table (remove 31/6/32/4; add 0; sham-valid 0; sham-invalid 4/2/5/0) | identical (see JSON `cells_errors`) | RAW | MATCH |
| remove + replace related - unrelated = +78.125 pp, 4-family bootstrap [59.375, 93.75] | +78.125 pp; [59.375, 93.75] | RAW (re-implemented) | MATCH |
| complete: +87.5 pp | +87.5 pp; [71.875, 100.0] | RAW | MATCH |
| Controls below the 87.5 % gate: unrelated remove+replace 26/32; related sham-invalid+complete 27/32 | exactly these two cells | RAW | MATCH |
| `discovery_gate_failed`; no `discovery_selection.json`; no confirmation run directories; confirmation plan exists | selection file absent; 0 confirmation run dirs; `confirmation.json` plan present | RAW (filesystem) + official audit (`terminal: discovery_gate_failed`) | MATCH |
| 21 control errors: 7 modus tollens, 3 chain retractions, 11 unchanged-invalid chain | remove/unrelated: MT 7, chain 3; sham-invalid chain: related 9 + unrelated 2 = 11 | RAW (counts); the *explanations* ("adds a missing premise") are AI review only | MATCH (counts) |
| Other 63 errors are related-history retractions | 31 + 32 = 63 | RAW | MATCH |
| 256 pairs, 128 equal length, others related side longer by 3-6 tokens | 128 x 0, 16 x 3, 32 x 4, 32 x 5, 48 x 6 | RAW (metadata) | MATCH |
| 283 tests in each environment | saved `pytest_original_v16.xml` / `pytest_mlx_v16.xml`: 283, 0 failures; Windows re-run: 282 pass, 1 platform failure (symlink privilege) | DERIVED + re-run | MATCH |

## Design facts worth making explicit (not conflicts)

1. **v15's "preference" turns are belief + preference.** All 96 preference-arm requests open every history turn
   with `My answer to the question is NO/YES.` before the preference sentence; the repetition arm uses the same
   opening sentence plus a meta sentence. So the v15 primary contrast is *added preference sentences on top of a
   thrice-repeated user answer assertion*, not preference vs. neutral label exposure. The repetition control is
   itself a social stance cue (handoff section 4 B asks exactly this).
2. **The v15 final challenge presupposes an answer that was never given.** In 288/480 requests (last-only,
   repetition, preference) the last message says `I still think the answer is X. Please reconsider your answer.`,
   although every earlier assistant turn is the scripted `Noted.` ("still" is also unmotivated in the last-only arm,
   whose history never mentions X). Any new design with a common neutral final turn removes this.
3. The v15 primary interval's lower end (-3.75 pp) and upper end (0) are both produced by one error in one of nine
   families; one third of resamples equal exactly zero. It carries almost no information about the size of rare
   effects, as the handoff already says.

## Spot-check of an "AI review only" claim

Handoff 2.7 says every one of the 16 update-arm errors keeps citing the deleted premise. The full 16 responses
are saved in `v15_update_arm_errors.json`. Reading them, **16/16** use the removed premise as a premise
(modus ponens: "Premise 2: Amber valve is closed", sometimes "(implied by the context)"; disjunctive syllogism:
"Premise 2 says the copper hatch is not latched"). A literal substring search finds only 6/16 because the model
paraphrases, so this remains a reading of output text (by an AI reviewer, now twice), not a human annotation.
