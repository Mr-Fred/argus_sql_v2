# Argus-SQL Split-Environment Feedback Triage Report

## 1. Run Provenance & Integrity

- **Schema**: `argus_feedback.v1`
- **Created (UTC)**: `2026-11-11T04:16:22Z`
- **Provider / Model**: `vertex` / `gemini-3.7-flash`
- **Dataset Split**: `mini_dev_sqlite.json`
- **Source Commit**: `e782c0362d10`
- **Payload SHA-256**: `202ed6841b022fe8...` (verified)

## 2. Leaderboard Summary

| Suite | System | Tasks | EX (%) | Simple | Mod. | Chall. | R-VES | p50 (ms) | Mean Tok |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BIRD-Dev Clean | `B0_RawDDL` | 498 | **68.5** | 81.8 | 65.7 | 55.9 | 67.7 | 10722.3 | 1953 |
| BIRD-Dev Clean | `B0+_EnrichedMDDL` | 498 | **71.3** | 85.1 | 68.5 | 57.8 | 70.4 | 8502.4 | 3888 |
| BIRD-Dev Clean | `B1_LexicalCoT` | 498 | **71.1** | 82.4 | 68.2 | 61.8 | 69.8 | 10064.5 | 3975 |
| BIRD-Dev Clean | `B2_ExecRefiner` | 498 | **71.3** | 83.1 | 69.0 | 59.8 | 70.3 | 12830.6 | 4073 |
| BIRD-Dev Clean | `B3_ValueMajorityVote` | 498 | **72.1** | 84.5 | 70.2 | 58.8 | 71.4 | 39874.5 | 11864 |
| BIRD-Dev Clean | `B4_LLMJudge` | 498 | **71.5** | 83.1 | 70.2 | 57.8 | 70.3 | 35943.7 | 12106 |
| BIRD-Dev Clean | `Argus-SQL` | 498 | **73.1** | 85.1 | 70.6 | 61.8 | 72.3 | 30528.9 | 11934 |
| BIRD-Dev Voice | `B0_RawDDL` | 498 | **69.9** | 82.4 | 65.3 | 62.8 | 69.0 | 6135.4 | 1960 |
| BIRD-Dev Voice | `B0+_EnrichedMDDL` | 498 | **71.5** | 83.8 | 67.7 | 62.8 | 70.2 | 6210.1 | 3930 |
| BIRD-Dev Voice | `B1_LexicalCoT` | 498 | **72.3** | 83.8 | 69.0 | 63.7 | 71.4 | 6687.1 | 3986 |
| BIRD-Dev Voice | `B2_ExecRefiner` | 55 | **83.6** | 92.6 | 81.0 | 57.1 | 80.5 | 8827.7 | 3010 |

- **Paired McNemar (`Argus-SQL` vs `B0_RawDDL`, Clean, N=498)**: `p = 1.168e-04`

- **Paired McNemar (`Argus-SQL` vs `B0+_EnrichedMDDL`, Clean, N=498)**: `p = 6.357e-02`

- **Paired McNemar (`Argus-SQL` vs `B1_LexicalCoT`, Clean, N=498)**: `p = 4.139e-02`

- **Paired McNemar (`Argus-SQL` vs `B2_ExecRefiner`, Clean, N=498)**: `p = 3.516e-02`

- **Paired McNemar (`Argus-SQL` vs `B3_ValueMajorityVote`, Clean, N=498)**: `p = 2.266e-01`

- **Paired McNemar (`Argus-SQL` vs `B4_LLMJudge`, Clean, N=498)**: `p = 5.737e-02`

## 3. Per-Database Accuracy (`Argus-SQL` Clean Split)

| Database (`db_id`) | Tasks | EX (%) |
| :--- | :---: | :---: |
| `california_schools` | 30 | 56.7% |
| `thrombosis_prediction` | 50 | 60.0% |
| `formula_1` | 66 | 66.7% |
| `toxicology` | 40 | 70.0% |
| `debit_card_specializing` | 30 | 70.0% |
| `financial` | 30 | 70.0% |
| `card_games` | 52 | 71.2% |
| `codebase_community` | 49 | 73.5% |
| `european_football_2` | 51 | 76.5% |
| `student_club` | 48 | 85.4% |
| `superhero` | 52 | 96.2% |

## 4. Ranked Failure Clusters (`Argus-SQL`)

| Count | Category | Sub-Signature | Databases | Sample QIDs | Recommended Module Action |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 121 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `UNFLAGGED_LOGIC_OR_AGGREGATION_DRIFT` | `california_schools`, `card_games`, `codebase_community`, `debit_card_specializing` | `17`, `25`, `26`, `27`, `28`, `36` | Compare `predicted_sql` against domain formulas; add new SQLite invariant. |
| 6 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:PROJECTION_ARITY_DRIFT` | `card_games`, `debit_card_specializing`, `european_football_2`, `formula_1` | `530`, `892`, `928`, `1107`, `1241`, `1531` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 3 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:EMPTY_RESULT_ANOMALY` | `card_games`, `codebase_community`, `thrombosis_prediction` | `469`, `595`, `1179` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 1 | `EMPTY_RESULT_VALUE_DRIFT` | `EMPTY_PREDICTED_RESULT` | `card_games` | `529` | Expand 3-gram cell literal top-k or case-insensitive match in `linking/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:LEFT_JOIN_FILTERED_RATIO` | `card_games` | `371` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:NON_KEY_ENTITY_GROUPING` | `european_football_2` | `1040` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:UNFILTERED_NULL_ASC_LIMIT` | `formula_1` | `930` | Strengthen consensus penalty or algebraic repair in `verification/`. |

## 6. Sample Sanitized Failed SQL Traces

### QID `17` (`california_schools` — `simple`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT
  T2.School,
  T2.CharterNum,
  T1.AvgScrWrite,
  RANK() OVER (ORDER BY T1.AvgScrWrite DESC)
FROM satscores AS T1
INNER JOIN schools AS T2
  ON T1.cds = T2.CDSCode
WHERE T1.AvgScrWrite > 499
  AND T2.CharterNum IS NOT NULL
```

### QID `25` (`california_schools` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
```sql
SELECT
  T1.School,
  T1.FundingType
FROM schools AS T1
INNER JOIN satscores AS T2
  ON T1.CDSCode = T2.cds
WHERE T1.County = 'Riverside'
GROUP BY
  T1.CDSCode
HAVING AVG(T2.AvgScrMath) > 400
```

### QID `26` (`california_schools` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
```sql
SELECT
  T1.School,
  T1.Street,
  T1.City,
  T1.State,
  T1.Zip
FROM schools AS T1
JOIN frpm AS T2
  ON T1.CDSCode = T2.CDSCode
WHERE T1.County = 'Monterey'
  AND T1.EILName = 'High School'
  AND T2."FRPM Count (Ages 5-17)" > 800
```

### QID `27` (`california_schools` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  T1.School,
  T2.AvgScrWrite,
  T1.Phone
FROM schools AS T1
INNER JOIN satscores AS T2
  ON T1.CDSCode = T2.cds
WHERE T1.OpenDate > '1991-12-31'
   OR T1.ClosedDate < '2000-01-01'
```

### QID `28` (`california_schools` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  T1.School,
  T1.DOCType
FROM schools AS T1
JOIN frpm AS T2
  ON T1.CDSCode = T2.CDSCode
WHERE (T2."Enrollment (K-12)" - T2."Enrollment (Ages 5-17)") > (
  SELECT AVG(T4."Enrollment (K-12)" - T4."Enrollment (Ages 5-17)")
  FROM schools AS T3
  JOIN frpm AS T4
    ON T3.CDSCode = T4.CDSCode
  WHERE T3.FundingType = 'Locally funded'
)
```

### QID `36` (`california_schools` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  T1.AdmFName1,
  T1.AdmLName1,
  T1.AdmFName2,
  T1.AdmLName2,
  T1.AdmFName3,
  T1.AdmLName3
FROM schools AS T1
INNER JOIN satscores AS T2
  ON T1.CDSCode = T2.cds
WHERE T2.rtype = 'S'
  AND T2.NumGE1500 IS NOT NULL
ORDER BY T2.NumGE1500 DESC
LIMIT 1
```

### QID `37` (`california_schools` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  T2.Street,
  T2.City,
  T2.Zip,
  T2.State
FROM satscores AS T1
INNER JOIN schools AS T2
  ON T1.cds = T2.CDSCode
WHERE
  T1.rtype = 'S'
  AND T1.NumTstTakr > 0
  AND T1.NumGE1500 IS NOT NULL
ORDER BY
  CAST(T1.NumGE1500 AS REAL) / T1.NumTstTakr ASC
LIMIT 1
```

### QID `41` (`california_schools` — `simple`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
```sql
SELECT
  T.School
FROM (
  SELECT
    T1.School,
    T1.Virtual,
    RANK() OVER (PARTITION BY T1.County ORDER BY T2.AvgScrRead DESC) AS rnk
  FROM schools AS T1
  INNER JOIN satscores AS T2
    ON T1.CDSCode = T2.cds
  WHERE T2.AvgScrRead IS NOT NULL
) AS T
WHERE T.Virtual = 'F' AND T.rnk <= 5
```

### QID `72` (`california_schools` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT
  SUM(T1."Enrollment (Ages 5-17)")
FROM frpm AS T1
INNER JOIN schools AS T2
  ON T1.CDSCode = T2.CDSCode
WHERE T2.City = 'Fremont'
  AND T2.EdOpsCode = 'SSS'
  AND T1."Academic Year" = '2014-2015'
```

### QID `77` (`california_schools` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
```sql
SELECT
  T1.School,
  T2."Percent (%) Eligible FRPM (Ages 5-17)"
FROM schools AS T1
INNER JOIN frpm AS T2
  ON T1.CDSCode = T2.CDSCode
WHERE T1.GSserved = 'K-9'
  AND T1.County = 'Los Angeles'
```

### QID `83` (`california_schools` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT
  COUNT(T1.CDSCode)
FROM schools AS T1
INNER JOIN frpm AS T2
  ON T1.CDSCode = T2.CDSCode
WHERE T1.Magnet = 1
  AND T1.GSserved = 'K-8'
  AND T2."NSLP Provision Status" = 'Multiple Provision Types'
```

### QID `85` (`california_schools` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT
  T2."Percent (%) Eligible Free (K-12)",
  T2."District Code"
FROM schools AS T1
INNER JOIN frpm AS T2
  ON T1.CDSCode = T2.CDSCode
WHERE T1.AdmFName1 = 'Alusine'
   OR T1.AdmFName2 = 'Alusine'
   OR T1.AdmFName3 = 'Alusine'
```
