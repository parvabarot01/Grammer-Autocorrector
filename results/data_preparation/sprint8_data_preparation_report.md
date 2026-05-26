# Sprint 8 Data Preparation Report

## Executive Summary

- Generated: `2026-05-26T09:44:10.328711+00:00`
- Input rows before cleaning: `2018`
- Clean rows after cleaning: `1947`
- Clean train / validation / test: `1560` / `195` / `192`
- Balanced train rows: `4572`
- BERT correct:error token ratio: `3.6124`

## Files Used

- `C:\Users\parva\OneDrive\Desktop\Project\Grammer Autocorrector\data\raw\grammar_correction.csv`
- `C:\Users\parva\OneDrive\Desktop\Project\Grammer Autocorrector\data\processed\grammar_correction_train.jsonl`
- `C:\Users\parva\OneDrive\Desktop\Project\Grammer Autocorrector\data\processed\grammar_correction_validation.jsonl`
- `C:\Users\parva\OneDrive\Desktop\Project\Grammer Autocorrector\data\processed\grammar_correction_test.jsonl`

## Cleaning Actions Performed

- Removed rows with empty or null original/corrected text.
- Normalized whitespace, unicode, and control characters.
- Preserved clean/no_error unchanged rows and flagged unchanged non-no_error rows.
- Removed exact duplicate source-target pairs.
- Removed split leakage by keeping the earliest pair in train, then validation, then test.
- Added normalized_error_type to all new clean exports.

## Rows Removed and Why

- Rows removed total: `3`
- `exact_duplicate_pair`: `2`
- `split_leakage_duplicate_kept_in_train`: `1`

## Rows Flagged and Why

- Rows flagged total: `68`
- `original_equals_corrected_without_no_error_label`: `68`

## Duplicate and Leakage Summary

- Exact duplicates removed: `2`
- Leakage duplicates removed: `1`
- Near duplicates detected before cleaning: `6`
- Near duplicates detected after cleaning: `6`

## Before / After Class Distribution

| normalized_error_type | before_full | after_clean_full | clean_train | balanced_train |
|---|---|---|---|---|
| article | 100 | 97 | 79 | 508 |
| capitalization | 40 | 39 | 31 | 508 |
| mixed_multiple | 438 | 407 | 327 | 508 |
| other | 649 | 636 | 508 | 508 |
| preposition | 95 | 95 | 76 | 508 |
| punctuation | 198 | 190 | 153 | 508 |
| spelling | 100 | 88 | 70 | 508 |
| subject_verb_agreement | 100 | 100 | 80 | 508 |
| tense | 298 | 295 | 236 | 508 |

## Before / After Dataset Stats

- Average original length before: `8.989` tokens
- Average corrected length before: `8.725` tokens
- Average original length after: `8.974` tokens
- Average corrected length after: `8.701` tokens

## Oversampling Strategy

- Oversampling enabled: `True`
- Majority class size used as target: `508`
- Effective target class size: `508`
- Validation and test splits were not oversampled.

### Oversample Counts

| normalized_error_type | oversampled_rows_added |
|---|---|
| article | 429 |
| capitalization | 477 |
| mixed_multiple | 181 |
| other | 0 |
| preposition | 432 |
| punctuation | 355 |
| spelling | 438 |
| subject_verb_agreement | 428 |
| tense | 272 |

## BERT Token-Label Summary

- Total examples: `1560`
- Total tokens: `14091`
- Correct tokens: `11036`
- Error tokens: `3055`
- Correct:error ratio: `3.6124`

## Training Readiness Conclusion

- The cleaned corpus is structurally ready for T5 and BERT preprocessing.
- Balanced training data now reduces class skew at the sentence-label level.
- BERT token labels are still weak labels derived from sentence diffs, so weighted loss is still recommended in the next sprint.

## Recommended Next Sprint

- Use the cleaned and balanced exports for T5 and BERT training only after reviewing the flagged noisy rows.
- Keep external datasets such as JFLEG or CoNLL on the roadmap to improve generalization beyond this relatively small corpus.
