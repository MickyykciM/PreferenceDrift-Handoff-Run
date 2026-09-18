# FINDINGS: independent recount of handoff sections 2.5 and 2.6 (v4 to v14)

- **Snapshot:** `_work/v16/project` (read-only; unmodified).
- **Script:** `recompute_v4_v14.py`. **Output:** `recompute_v4_v14.json` (its `claims` array holds all 66 rows).

**Basis legend:**
- **RAW**: recomputed from `generations/*.json`, joined by exact messages to the frozen design. Gold labels were re-derived from the prompt text or the formal certificates, with 0 mismatches.
- **DERIVED**: taken only from a saved summary, audit or record.

## CONFLICTS
None. No correction proposed.

## NOT VERIFIABLE
| Claim | Reason |
|---|---|
| 2.5 "168 项测试通过" (1.5B) | No saved test record for v4; stated only in the report and README. The snapshot's suite is v16's |
| 2.5 "187 项测试通过" (3B) | Same |

## The 2,879 + 2 convention

**Rule that reproduces the handoff:** count one per completed checkpoint (`status == "completed"`) in the 21 v5–v14 run directories:
80 + 34 + 0 + 64 + 232 + 96 + 96 + 192 + 192 + 192 + 96 + 165 + 288 + 288 + 288 + 96 + 384 + 96 = 2,879.

**The 2 unfinished** are the two checkpoints with `status == "running"` (the 3B format run and v10).

**What this count does and does not do:**
- Aliases share a call: the 7B regression has 232 calls for 264 records.
- It includes 25 truncated calls and 199 calls from interrupted runs.
- It matches the project's saved audit counters at every stage.
- It is not a count of independent items: only 1,720 distinct requests occur among the 2,879.

**Other counting conventions:**

| Convention | Total |
|---|---|
| Completed checkpoints (the handoff's count) | 2,879 |
| Records instead of calls | 2,911 |
| Adding the 2 unfinished checkpoints | 2,881 |
| Excluding the two interrupted runs | 2,680 |
| Planned requests | 3,032 |

**Outcomes of the 2,879:** 2,329 correct, 188 wrong, 25 truncated, 337 unknown.

## Claim table
| Sec | Claim | Recomputed | Source | Basis | Status | Note |
|---|---|---|---|---|---|---|
| 2.5 | 264 条条件记录→232 次生成；32 别名 (×2) | 264/232/232/32 both models | v4 runs cases.json + generations | RAW | MATCH | alias groups 200×1, 32×2 |
| 2.5 | 1.5B 删去论证 8/8 vs 6/8 | 8/8, 6/8 | 1.5B v4 | RAW | MATCH | |
| 2.5 | 1.5B 新增真假 14/16 vs 8/16 | 14/16, 8/16 | 1.5B v4 | RAW | MATCH | |
| 2.5 | 1.5B 80 个单项有效性全部 INVALID | 80/80 INVALID | 1.5B v4 | RAW | MATCH | one joint answer was VALID (outside claim) |
| 2.5 | 1.5B 复制／加法 8/8 | 8/8 | 1.5B v4 | RAW | MATCH | |
| 2.5 | 1.5B 两轴筛查失败 | truth fail, validity fail | 1.5B v4 | RAW | MATCH | saved summary: 17 cells and 14 contrasts equal |
| 2.5 | 168 项测试 | – | report | NOT VERIFIABLE | NOT VERIFIABLE | |
| 2.5 | 3B 同请求复核 | 232/232 identical (messages, seed, decoding) | both v4 | RAW | MATCH | |
| 2.5 | 3B 新增真假 15/16 vs 12/16 | 15/16, 12/16 | 3B v4 | RAW | MATCH | |
| 2.5 | 3B 73 INVALID / 7 VALID of 80 | 73/7 (YES-first 33/7, NO-first 40/0) | 3B v4 | RAW | MATCH | |
| 2.5 | 3B 16/16 VALID→16/16 INVALID；两项同时正确 5/16 | same; 5/16, 5/16 | 3B v4 | RAW | MATCH | |
| 2.5 | 3B 两轴仍失败 | both fail | 3B v4 | RAW | MATCH | |
| 2.5 | 187 项测试 | – | report | NOT VERIFIABLE | NOT VERIFIABLE | |
| 2.5 | 两次均全部 EOS | 232/232 eos each | both v4 | RAW | MATCH | |
| 2.5 | 同系列两个检查点 | Qwen2.5-1.5B / Qwen2.5-3B-Instruct | metadata | RAW | MATCH | |
| 2.5 src | Both reports' cell tables, joint tables, order flips, context pairs, cross-model transitions | all equal | v4 reports vs runs | RAW | MATCH | see JSON |
| 2.5 src | raw sha256 1301bace… / d5c4ac1c… | match | samples.json | RAW | MATCH | |
| 2.6 | 2,879 完成 + 2 未完成 | 2,879 + 2 | all v5–v14 generations | RAW | MATCH | see convention above |
| 2.6 | 不包括 v1–v4、v15–v16 | exactly the 21 non-v3/v4 20260914_* dirs | results/runs | RAW | MATCH | |
| 2.6 | 1.5B 四种配置 80；三配置文本一致；FP16 eager 全截断 | 80; 20/20 identical (and = v4 text); eager 20/20 length @96, all "!" | runtime_v5_* | RAW | MATCH | exact-correct 11/20 in each good profile |
| 2.6 | 3B BF16 34/64 + 1 未完成 | 34 completed, 1 running, 64 planned; bf16 | format_development_qwen3b_v5 | RAW (cause DERIVED) | MATCH | memory-pressure cause from interruption note |
| 2.6 | 7B MLX 首次加载 0 | no generations/; generation_calls 0 | *_load_failure | RAW (cause DERIVED) | MATCH | |
| 2.6 | 7B 筛查 64；direct、reasoning 16/16；选 direct | 64; 16/16, 16/16 (fewshot 15/16 ×2); first pass = direct | mlx_qwen7b_screen_v5 | RAW | MATCH | saved selection = direct |
| 2.6 | 7B 回归 232；两轴失败 | 232 calls / 264 records; both fail | mlx_qwen7b_regression_v5 | RAW | MATCH | |
| 2.6 | v6 开发 94/96 通过 | 94/2; pass | clean_direct_qwen7b_dev_v6 | RAW | MATCH | |
| 2.6 | v6 第一批 83/96 失败 | 83/13; truth 20/24 ×2 | clean_direct_qwen7b_holdout_v6 | RAW | MATCH | |
| 2.6 | v7 22 对 4 错 166 未知 | 22/4/166, 0 truncated; fail | reasoning_qwen7b_dev_v7 | RAW | MATCH | 159/166 = ANSWER on explanation line |
| 2.6 | v8 180/192 分组/顺序失败 | 180/12, 0 unknown; opened_v5 validity NO-first 20/24, flips 4/24 | guided_qwen7b_dev_v8 | RAW | MATCH | |
| 2.6 | v9 开发 190/192 通过 | 190/2 (holdout_reverse_chain_0 ×2); pass | guided_qwen3_4b_dev_v9 | RAW | MATCH | |
| 2.6 | v9 第二批 94/96，2 同一减法题循环截断，未知门槛失败 | 94 + 2 truncated ("32 - 9 = 23" ×2, 256 tok); only the unknown-pair check fails | guided_qwen3_4b_holdout_v9 | RAW | MATCH | |
| 2.6 | v10 165/288 + 1 未完成 | 165 completed, 1 running, 288 planned; all constraint_steps = 3 vs scorer's 2 | constrained_qwen3_4b_dev_v10 | RAW (mechanism DERIVED) | MATCH | no accuracy claimed |
| 2.6 | v11 266/288 无未知 失败 | 266/22, 0 unknown; opened_v7 validity 18/24 ×2 | constrained_qwen3_4b_dev_v11 | RAW | MATCH | see Observation 1 |
| 2.6 | v12 272/288：2 截断、3 真假错、11 有效性错 | 272 / 2 truncated (truth) / 3 / 11; fail | axis_guided_qwen3_4b_dev_v12 | RAW | MATCH | |
| 2.6 | v13 开发 284/288 通过 | 284/4 (holdout_reverse_chain_0, _2 ×2); pass | guided_qwen3_4b_8bit_dev_v13 | RAW | MATCH | |
| 2.6 | v13 第三批 94/96，1 有效性错，1 减法截断，未知门槛失败 | 94 / wrong v10_reverse_chain_2 YES-first / truncated "47 - 18 = 29" NO-first | guided_qwen3_4b_8bit_holdout_v13 | RAW | MATCH | |
| 2.6 | v14 开发 378/384 零未知 全通过 | 378/6/0; pass | repetition_qwen3_8bit_dev_v14 | RAW | MATCH | |
| 2.6 | v14 第四批 94/96 零未知 全通过 | 94/2/0; pass | repetition_qwen3_8bit_holdout_v14 | RAW | MATCH | |
| 2.6 | row models | 7B = Qwen2.5-7B-Instruct-4bit (v5–v8); Qwen3-4B-2507 4-bit (v9–v12); 8-bit (v13–v14); penalty 1.1/64 | checkpoint model blocks | RAW | MATCH | HF 114, MLX 2,765, fake 0 |
| 2.6 | v11 容差修订，标签/终止/准确率/失败不变 | 13/288 pairs > .001; max .0031890869140625 ≤ 2^-7; 288/288 traces and argmax ok; labels as stored; still fails | v11 metadata | RAW | MATCH | |
| 2.6 | v6 元组/列表、v7 格式问题保留，未改分 | stored scores = strict re-parse (v6 192, v7 192); v6 validation hash = pre-fix 2dae1aa7…; 24 list certificates | v6/v7 runs | RAW | MATCH | |
| 2.6 | v13 4/8-bit 同 288 请求均 284 | identical requests; 284 vs 284; 282 c-c, 2 trunc-c, 2 c-w, 2 w-w | v9 dev+val vs v13 dev | RAW | MATCH | |
| 2.6 | v14 vs v13 同 384 均 378；一截断变对、一对变错 | 377 c-c, 1 trunc-c (v10_integer_subtraction_0_1), 1 c-w (v10_reverse_chain_0), 5 w-w | v13 dev+val vs v14 dev | RAW | MATCH | |
| 2.6 | v14 开发 真假 192/192 有效性 186/192 | same | v14 dev | RAW | MATCH | |
| 2.6 | 第四批 24/24 ×2；22/24、24/24；翻转 2/24 | same (v14_reverse_chain_1, _2) | v14 validation | RAW | MATCH | |
| 2.6 | 四批验证都已打开 | v5 bank←v6, v7←v9, v10←v13, v14←v14; later dev sets include opened_v5/v7/v10 | validation manifests | RAW | MATCH | |
| 2.6 | joint_task_validated / preference_effect_established / publication_ready = false | all false | measurement_repair_v14_final_audit.json | DERIVED | MATCH | recount agrees the v14 gates pass |
| 2.6 | 两套环境各 269 测试 | 269 testcases each, 0 fail/error/skip; receipt sha256 match | pytest_{mlx,original}_v14.xml | DERIVED | MATCH | not re-run |
| 2.6 | 原始哈希、零调用恢复 | v14 hashes b2171414… / 72774ed3… match; new_calls 0 | samples.json, manifests | RAW (hash) / DERIVED | MATCH | resume not recorded for runtime ×4, 7B screen, 7B regression |
| 2.6 src | v6 truth 20/24 ×2; v8 20/24 and 4/24; v9 dev table; v11 18/24 ×2; v13 dev 144/144 and 140/144; v13 val 24/24, 23/24 / 23/24, 24/24; v14 dev 6 reverse-chain errors, flips only opened_v10 2/24; v14 val INVALID YES-first 10/12 | all equal | MEASUREMENT_REPAIR_REPORT vs runs | RAW | MATCH | |
| 2.6 src | cumulative 1,727 (+2), 2,015, 2,399, 2,879 | equal; 8/8 saved counters equal | analyses/*audit.json | RAW | MATCH | |
| 2.6 src | v14 timeline 03:07:00, 03:08:59, 05:03:28, 05:03:31 UTC | equal | commitment, lock, manifests | DERIVED | MATCH | self-recorded |
| 2.6 src | 15 doc/audit raw sha256 values | 15/15 match | samples.json | RAW | MATCH | |

## Observations (no handoff number changes)

1. **Negation-scope wording in the v7/v10/v14 banks.** "it is not the case that (A) or (B)" can be read as NOT(A or B), which is the gold, or as (NOT A) or B.
   - v11's gate failure rests entirely on its 6 demorgan_disjunction answers (all INVALID).
   - Counterfactual, not a re-score: treating those 6 as correct gives opened_v7 validity 21/24 ×2 (INVALID 9/12), and v11 would pass.
   - Every other version answered these items correctly. v11's outputs are bare YES/NO, so a misreading cannot be told apart from a logic error.
2. **v7 unknowns.** 159 put a single final ANSWER on the explanation line, 6 had the marker not at the end, and 1 had no marker. Informational only: under the later v8 rule, 159 of them would parse, and 153 of those are correct.
3. **The 25 truncations are loops.** 20 are FP16-eager outputs made only of "!". 4 repeat "32 - 9 = 23" (v9 ×2, v12 ×2), and 1 repeats "47 - 18 = 29" (v13).

## Method and limits
- Standard library plus numpy only. The script imports and executes nothing from the project, and refuses to write inside the project.
- Parsers and gates are new code implementing each version's frozen rule, so they share the rule definitions by design.
- Cross-checks against stored scores, saved summaries and audit counters all agree.
- Certificate-based gold assumes each certificate matches its prompt text; Observation 1 is the one ambiguous wording found.
- The project's own `audit_measurement_v14.py` was re-run separately through the Windows adapter (see `outputs/`). It independently reproduces 2,879 + 2.
