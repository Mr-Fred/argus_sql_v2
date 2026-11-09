# Argus-SQL Split-Environment Feedback Triage Report

## 1. Run Provenance & Integrity

- **Schema**: `argus_feedback.v1`
- **Created (UTC)**: `2026-11-09T13:51:01Z`
- **Provider / Model**: `vertex` / `gemini-3.7-flash`
- **Dataset Split**: `mini_dev_sqlite.json`
- **Source Commit**: `9b9e0c571c81`
- **Payload SHA-256**: `59b2c3f6429603b0...` (verified)

## 2. Leaderboard Summary

| Suite | System | Tasks | EX (%) | Simple | Mod. | Chall. | R-VES | p50 (ms) | Mean Tok |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BIRD-Dev Clean | `B0_RawDDL` | 110 | **70.9** | 81.1 | 73.0 | 58.3 | 71.6 | 5238.2 | 1926 |
| BIRD-Dev Clean | `B0+_EnrichedMDDL` | 110 | **72.7** | 81.1 | 75.7 | 61.1 | 73.0 | 5869.0 | 3647 |
| BIRD-Dev Clean | `B1_LexicalCoT` | 110 | **72.7** | 83.8 | 75.7 | 58.3 | 73.6 | 6329.8 | 3632 |
| BIRD-Dev Clean | `B2_ExecRefiner` | 110 | **73.6** | 83.8 | 81.1 | 55.6 | 73.6 | 5730.7 | 2889 |
| BIRD-Dev Clean | `B3_ValueMajorityVote` | 110 | **77.3** | 81.1 | 86.5 | 63.9 | 78.0 | 6207.4 | 11416 |
| BIRD-Dev Clean | `B4_LLMJudge` | 110 | **76.4** | 83.8 | 86.5 | 58.3 | 77.0 | 8125.7 | 11783 |
| BIRD-Dev Clean | `Argus-SQL` | 110 | **75.5** | 83.8 | 83.8 | 58.3 | 76.4 | 12962.5 | 11532 |
| BIRD-Dev Voice | `B0_RawDDL` | 110 | **69.1** | 78.4 | 73.0 | 55.6 | 69.5 | 5793.3 | 1944 |
| BIRD-Dev Voice | `B0+_EnrichedMDDL` | 110 | **73.6** | 81.1 | 78.4 | 61.1 | 73.9 | 6599.6 | 3650 |
| BIRD-Dev Voice | `B1_LexicalCoT` | 110 | **70.9** | 78.4 | 75.7 | 58.3 | 70.9 | 6763.9 | 3622 |
| BIRD-Dev Voice | `B2_ExecRefiner` | 110 | **70.9** | 78.4 | 78.4 | 55.6 | 70.0 | 4234.7 | 2307 |
| BIRD-Dev Voice | `B3_ValueMajorityVote` | 110 | **74.5** | 83.8 | 81.1 | 58.3 | 75.0 | 7259.7 | 11514 |
| BIRD-Dev Voice | `B4_LLMJudge` | 110 | **72.7** | 81.1 | 78.4 | 58.3 | 73.2 | 9549.6 | 11945 |
| BIRD-Dev Voice | `Argus-SQL` | 110 | **75.5** | 83.8 | 81.1 | 61.1 | 75.9 | 4458.9 | 6836 |

- **Paired McNemar (`Argus-SQL` vs `B0_RawDDL`, Clean, N=110)**: `p = 2.668e-01`

- **Paired McNemar (`Argus-SQL` vs `B0+_EnrichedMDDL`, Clean, N=110)**: `p = 5.078e-01`

- **Paired McNemar (`Argus-SQL` vs `B1_LexicalCoT`, Clean, N=110)**: `p = 4.531e-01`

- **Paired McNemar (`Argus-SQL` vs `B2_ExecRefiner`, Clean, N=110)**: `p = 6.250e-01`

- **Paired McNemar (`Argus-SQL` vs `B3_ValueMajorityVote`, Clean, N=110)**: `p = 6.250e-01`

- **Paired McNemar (`Argus-SQL` vs `B4_LLMJudge`, Clean, N=110)**: `p = 1.000e+00`

## 3. Per-Database Accuracy (`Argus-SQL` Clean Split)

| Database (`db_id`) | Tasks | EX (%) |
| :--- | :---: | :---: |
| `california_schools` | 10 | 60.0% |
| `card_games` | 10 | 60.0% |
| `financial` | 10 | 60.0% |
| `toxicology` | 10 | 70.0% |
| `codebase_community` | 10 | 80.0% |
| `debit_card_specializing` | 10 | 80.0% |
| `european_football_2` | 10 | 80.0% |
| `formula_1` | 10 | 80.0% |
| `thrombosis_prediction` | 10 | 80.0% |
| `student_club` | 10 | 90.0% |
| `superhero` | 10 | 90.0% |

## 4. Ranked Failure Clusters (`Argus-SQL`)

| Count | Category | Sub-Signature | Databases | Sample QIDs | Recommended Module Action |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 22 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `UNFLAGGED_LOGIC_OR_AGGREGATION_DRIFT` | `california_schools`, `card_games`, `codebase_community`, `debit_card_specializing` | `586`, `94`, `847`, `197`, `198`, `28` | Compare `predicted_sql` against domain formulas; add new SQLite invariant. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:INTEGER_DIVISION_TRUNCATION` | `california_schools` | `24` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:JOIN_FANOUT_DUPLICATES` | `card_games` | `344` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:LEFT_JOIN_FILTERED_RATIO` | `card_games` | `371` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:PROJECTION_ARITY_DRIFT` | `financial` | `159` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:UNFILTERED_NULL_ASC_LIMIT` | `thrombosis_prediction` | `1168` | Strengthen consensus penalty or algebraic repair in `verification/`. |

## 5. Spoken Voice-BIRD Drift Analysis

- **QID `866` (`formula_1`, `moderate`)**: Clean EX=`True` -> Voice EX=`False` (`SILENT_SEMANTIC_OR_FORMULA_ERROR`)
- **QID `371` (`card_games`, `challenging`)**: Clean EX=`False` -> Voice EX=`True` (`CLEAN_MATCH`)

## 6. Sample Sanitized Failed SQL Traces

### QID `344` (`card_games` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal+argus_repair` | **Invariants**: `JOIN_FANOUT_DUPLICATES`
```sql
SELECT DISTINCT T1.name
FROM cards AS T1
INNER JOIN legalities AS T2
  ON T1.uuid = T2.uuid
WHERE T1.rarity = 'mythic'
  AND T2.format = 'gladiator'
  AND T2.status = 'Banned'
```

### QID `586` (`codebase_community` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
```sql
SELECT
  T3.DisplayName
FROM votes AS T1
INNER JOIN posts AS T2
  ON T1.PostId = T2.Id
INNER JOIN users AS T3
  ON T1.UserId = T3.Id
WHERE
  T1.BountyAmount = 50
  AND T2.Title LIKE '%variance%'
```

### QID `94` (`financial` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  T1.account_id,
  (SELECT MAX(A11) FROM district) - T3.A11 AS salary_gap
FROM disp AS T1
INNER JOIN client AS T2
  ON T1.client_id = T2.client_id
INNER JOIN district AS T3
  ON T2.district_id = T3.district_id
WHERE T2.gender = 'F'
ORDER BY T2.birth_date ASC, T3.A11 ASC
LIMIT 1
```

### QID `847` (`formula_1` — `simple`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT
  T2.surname
FROM qualifying AS T1
INNER JOIN drivers AS T2
  ON T1.driverId = T2.driverId
WHERE T1.raceId = 19
  AND T1.q2 IS NOT NULL
  AND T1.q2 != ''
ORDER BY T1.q2 ASC
LIMIT 1
```

### QID `197` (`toxicology` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  AVG(oxygen_count)
FROM (
  SELECT
    T1.molecule_id,
    COUNT(T2.atom_id) AS oxygen_count
  FROM (
    SELECT DISTINCT
      molecule_id
    FROM bond
    WHERE bond_type = '-'
  ) AS T1
  LEFT JOIN atom AS T2
    ON T1.molecule_id = T2.molecule_id
    AND T2.element = 'o'
  GROUP BY
    T1.molecule_id
)
```

### QID `371` (`card_games` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `LEFT_JOIN_FILTERED_RATIO`
```sql
SELECT
  CAST(COUNT(DISTINCT CASE WHEN T2.language = 'French' THEN T1.id END) AS REAL) * 100.0 / NULLIF(COUNT(DISTINCT T1.id), 0)
FROM cards AS T1
LEFT JOIN foreign_data AS T2
  ON T1.uuid = T2.uuid
WHERE T1.isStorySpotlight = 1
```

### QID `198` (`toxicology` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  CAST(SUM(CASE WHEN T2.bond_type = '-' THEN 1 ELSE 0 END) AS REAL) / NULLIF(COUNT(T3.atom_id), 0)
FROM molecule AS T1
INNER JOIN bond AS T2
  ON T1.molecule_id = T2.molecule_id
INNER JOIN connected AS T3
  ON T2.bond_id = T3.bond_id
WHERE T1.label = '+'
```

### QID `28` (`california_schools` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
```sql
SELECT
  T2.School,
  T2.DOCType
FROM frpm AS T1
INNER JOIN schools AS T2
  ON T1.CDSCode = T2.CDSCode
WHERE (T1."Enrollment (K-12)" - T1."Enrollment (Ages 5-17)") > (
  SELECT
    AVG(T3."Enrollment (K-12)" - T3."Enrollment (Ages 5-17)")
  FROM frpm AS T3
  INNER JOIN schools AS T4
    ON T3.CDSCode = T4.CDSCode
  WHERE T4.FundingType = 'Locally funded'
)
```

### QID `340` (`card_games` — `simple`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT
  name
FROM cards
WHERE
  cardKingdomFoilId IS NOT NULL
  AND cardKingdomId IS NOT NULL
```

### QID `1168` (`thrombosis_prediction` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal+argus_repair` | **Invariants**: `UNFILTERED_NULL_ASC_LIMIT`
```sql
SELECT
  T2.Date,
  strftime('%Y', T1."First Date") - strftime('%Y', T1.Birthday)
FROM Patient AS T1
INNER JOIN Laboratory AS T2
  ON T1.ID = T2.ID
WHERE T1.Diagnosis = 'SJS' AND T1.Birthday IS NOT NULL ORDER BY T1.Birthday ASC
LIMIT 1
```

### QID `1029` (`european_football_2` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT
  MAX(buildUpPlaySpeed)
FROM Team_Attributes
GROUP BY
  team_api_id
ORDER BY
  MAX(buildUpPlaySpeed) DESC
LIMIT 4
```

### QID `115` (`financial` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
```sql
SELECT
  100.0 * SUM(CASE WHEN gender = 'M' THEN 1 ELSE 0 END) / COUNT(*)
FROM client
WHERE district_id = (
  SELECT district_id
  FROM district
  WHERE A3 = 'south Bohemia'
  ORDER BY CAST(A4 AS INTEGER) DESC
  LIMIT 1
)
```
