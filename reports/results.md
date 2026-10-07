# Results (test users, n=19,682)

Model: `lgbm-lambdarank-abe0bcb6`. Ground truth: every product in the user's held-out last order (including first-time purchases).

| System | recall@10 | precision@10 | ndcg@10 | map@10 | hit rate@10 | recall@20 | precision@20 | ndcg@20 | map@20 | hit rate@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) Global popularity | 0.0706 | 0.0727 | 0.0988 | 0.0447 | 0.4604 | 0.0957 | 0.0511 | 0.0983 | 0.0393 | 0.5240 |
| (b) User's most frequent items | 0.3299 | 0.2734 | 0.3965 | 0.2710 | 0.8518 | 0.4382 | 0.1973 | 0.4133 | 0.2650 | 0.9015 |
| (c) Retrieval only (retrieval score) | 0.3423 | 0.2864 | 0.4141 | 0.2890 | 0.8561 | 0.4483 | 0.2042 | 0.4283 | 0.2814 | 0.9000 |
| (d) Two-stage: retrieval + LGBM lambdarank | **0.3608** | **0.3032** | **0.4381** | **0.3108** | **0.8727** | **0.4673** | **0.2140** | **0.4506** | **0.3018** | **0.9119** |

## Candidate generation

| Metric | Value |
|---|---:|
| Candidate recall (all sources, upper bound for the ranker) | 0.6164 |
| Avg candidates per user | 98.0 |
| Candidate recall, `src_history` only | 0.5895 |
| Candidate recall, `src_i2v` only | 0.1333 |
| Candidate recall, `src_aisle` only | 0.1349 |
| Candidate recall, `src_global` only | 0.0886 |
| Share of truth items never bought before by the user | 0.4047 |

## Is the gain real? (paired bootstrap over test users, 1,000 resamples)

| Difference | Mean | 95% CI |
|---|---:|---:|
| (d) minus (b) - recall@10 | +0.0309 | [+0.0292, +0.0327] |
| (d) minus (b) - ndcg@10 | +0.0416 | [+0.0399, +0.0431] |
| (d) minus (c) - recall@10 | +0.0186 | [+0.0171, +0.0200] |
| (d) minus (c) - ndcg@10 | +0.0241 | [+0.0227, +0.0254] |
