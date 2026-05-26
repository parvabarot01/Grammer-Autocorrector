# Bias Report

- Generated: `2026-05-26T09:17:42.923106+00:00`

## Gender Pronoun Balance

| group | count |
|---|---|
| male | 272 |
| female | 330 |
| neutral | 144 |

## Profession Co-occurrence Snapshot

| profession | male | female | neutral | total |
|---|---|---|---|---|
| doctor | 6 | 6 | 0 | 12 |
| teacher | 4 | 1 | 4 | 9 |
| manager | 2 | 1 | 3 | 6 |
| lawyer | 2 | 2 | 0 | 4 |
| scientist | 2 | 0 | 2 | 4 |
| police | 1 | 0 | 0 | 1 |
| assistant | 1 | 0 | 0 | 1 |
| nurse | 0 | 1 | 0 | 1 |
| officer | 1 | 0 | 0 | 1 |
| student | 0 | 1 | 0 | 1 |

## Sentence Length Bias

- Mean original length: `9.019`
- P95 original length: `14.0`
- Max original length: `22`

## Vocabulary Repetition Bias

- Type-token ratio: `0.1798`

| token | count |
|---|---|
| new | 92 |
| than | 80 |
| if | 56 |
| store | 51 |
| book | 51 |
| very | 46 |
| more | 46 |
| car | 45 |
| work | 44 |
| will | 43 |
| not | 42 |
| students | 41 |
| movie | 41 |
| any | 41 |
| cake | 38 |

## Dominant Correction Patterns

| pattern | count | percentage |
|---|---|---|
| structural-rewrite | 391 | 19.38% |
| lexical-substitution | 364 | 18.04% |
| single-token-replacement | 330 | 16.35% |
| deletion | 315 | 15.61% |
| insertion | 172 | 8.52% |
| multi-token-rewrite | 157 | 7.78% |
| punctuation-only | 99 | 4.91% |
| abbreviation-expansion | 82 | 4.06% |
| unchanged | 68 | 3.37% |
| casing-only | 40 | 1.98% |

## Duplicate and Noise Bias

| indicator | value |
|---|---|
| Male pronouns | 272 |
| Female pronouns | 330 |
| Neutral pronouns | 144 |
| Exact pair duplicates | 3 |
| Near-duplicate pairs | 6 |
| Noisy rows | 70 |

## Assessment

- Dominant pronoun group: `female`
- Dominant correction pattern: `structural-rewrite`
- Duplicate risk: `moderate`
- Length bias risk: `low`
