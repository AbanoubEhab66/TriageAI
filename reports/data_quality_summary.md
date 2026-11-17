# Data Quality Summary - TriageAI Phase 1

## Record counts
| Step | Records |
|---|---|
| Raw records | 28587 |
| English records | 16338 |
| Removed (no classifiable text) | 0 |
| Final records | 16338 |

## Missing / empty values (English subset)
| Item | Count |
|---|---|
| Missing subject (null) | 2607 |
| Empty subject after strip | 2607 |
| Missing body (null) | 0 |
| Empty body after strip | 0 |
| Empty text records | 0 |

## Exact duplicates
| check | extra_copies | tickets_involved | duplicate_groups |
|---|---|---|---|
| raw text (subject+body) | 0 | 0 | 0 |
| normalised text | 1 | 2 | 1 |
| normalised body only | 19 | 38 | 19 |

Exact-duplicate groups with conflicting queues: 0

## Near duplicates
- Method: cosine similarity on TF-IDF (unigram+bigram, English stop words removed, max_df=0.5, min_df=2, sublinear tf)
- Threshold: cosine >= 0.7 - **CONFIRMED**
- Duplicate/similarity groups with more than one ticket: 2910
- Tickets in those groups: 7087 (near-duplicate copies beyond the first: 4177)
- Largest group: 7 tickets
- Groups with conflicting queues: 11 (33 tickets)

Threshold sweep:

| threshold | similar_pairs | pairs_with_different_queue | tickets_in_groups | groups_size>1 | largest_group | groups_with_label_conflict | % pairs different queue |
|---|---|---|---|---|---|---|---|
| 0.7 | 5192 | 15 | 7087 | 2910 | 7 | 11 | 0.3 |
| 0.8 | 2405 | 4 | 3689 | 1590 | 5 | 3 | 0.2 |
| 0.85 | 1279 | 2 | 2099 | 938 | 5 | 1 | 0.2 |
| 0.9 | 533 | 2 | 917 | 424 | 5 | 1 | 0.4 |
| 0.95 | 147 | 2 | 265 | 127 | 4 | 1 | 1.4 |
| 0.99 | 43 | 2 | 83 | 41 | 3 | 1 | 4.7 |

## Class distribution
| queue | tickets | percent |
|---|---|---|
| Technical Support | 4737 | 28.99 |
| Product Support | 3073 | 18.81 |
| Customer Service | 2410 | 14.75 |
| IT Support | 1942 | 11.89 |
| Billing and Payments | 1595 | 9.76 |
| Returns and Exchanges | 820 | 5.02 |
| Service Outages and Maintenance | 664 | 4.06 |
| Sales and Pre-Sales | 513 | 3.14 |
| Human Resources | 348 | 2.13 |
| General Inquiry | 236 | 1.44 |

Largest queue: Technical Support (4737); smallest: General Inquiry (236); imbalance ratio 20.1 : 1

## Actual split ratios
| Split | Actual samples | Actual % |
|---|---|---|
| Train | 11498 | 70.38 |
| Val | 2420 | 14.81 |
| Test | 2420 | 14.81 |

| queue | train_n | val_n | test_n | train_% | val_% | test_% | overall_% |
|---|---|---|---|---|---|---|---|
| Technical Support | 3377 | 667 | 693 | 29.37 | 27.56 | 28.64 | 28.99 |
| Product Support | 2126 | 466 | 481 | 18.49 | 19.26 | 19.88 | 18.81 |
| Customer Service | 1710 | 353 | 347 | 14.87 | 14.59 | 14.34 | 14.75 |
| IT Support | 1373 | 273 | 296 | 11.94 | 11.28 | 12.23 | 11.89 |
| Billing and Payments | 1125 | 249 | 221 | 9.78 | 10.29 | 9.13 | 9.76 |
| Returns and Exchanges | 569 | 128 | 123 | 4.95 | 5.29 | 5.08 | 5.02 |
| Service Outages and Maintenance | 454 | 103 | 107 | 3.95 | 4.26 | 4.42 | 4.06 |
| Sales and Pre-Sales | 347 | 86 | 80 | 3.02 | 3.55 | 3.31 | 3.14 |
| Human Resources | 256 | 55 | 37 | 2.23 | 2.27 | 1.53 | 2.13 |
| General Inquiry | 161 | 40 | 35 | 1.4 | 1.65 | 1.45 | 1.44 |

## Leakage check
- Groups crossing splits: 0
- Identical texts crossing splits: 0
| threshold | similar_pairs | pairs_crossing_splits | note |
|---|---|---|---|
| 0.7 | 5192 | 0 | must be 0 |
| 0.8 | 2405 | 0 | must be 0 |
| 0.85 | 1279 | 0 | must be 0 |
| 0.9 | 533 | 0 | must be 0 |
| 0.95 | 147 | 0 | must be 0 |
| 0.99 | 43 | 0 | must be 0 |

Plain stratified split vs group-aware split:

| threshold | plain split: % test with near-dup in train | plain split: % val with near-dup in train | group-aware: % test with near-dup in train | group-aware: % val with near-dup in train |
|---|---|---|---|---|
| 0.7 | 32.56 | 33.33 | 0.0 | 0.0 |
| 0.8 | 16.24 | 17.58 | 0.0 | 0.0 |
| 0.9 | 3.75 | 4.41 | 0.0 | 0.0 |
| 0.95 | 1.06 | 1.18 | 0.0 | 0.0 |

## Decisions
- Kept English tickets only (16,338 of 28,587).
- Tickets with empty subject are kept if the body is usable; tickets with no text at all are removed (0 removed).
- No ticket was deleted for being a duplicate; duplicates are grouped and kept together in one split.
- Groups with conflicting queues are flagged (has_label_conflict) and reported, not removed.
- Excluded answer/type/priority/tags/version to avoid leakage; priority is not risk.
- Split: StratifiedGroupKFold(20 folds, random_state=42) -> 14/3/3 folds; actual ratios reported below.