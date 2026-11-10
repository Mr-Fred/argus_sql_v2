# Argus-SQL Split-Environment Feedback Triage Report

## 1. Run Provenance & Integrity

- **Schema**: `argus_feedback.v1`
- **Created (UTC)**: `2026-11-10T07:14:44Z`
- **Provider / Model**: `vertex` / `gemini-3.7-flash`
- **Dataset Split**: `mini_dev_sqlite.json`
- **Source Commit**: `e39bbbc26bf2`
- **Payload SHA-256**: `c196f5ed3004e800...` (verified)

## 2. Leaderboard Summary

| Suite | System | Tasks | EX (%) | Simple | Mod. | Chall. | R-VES | p50 (ms) | Mean Tok |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BIRD-Dev Clean | `B0_RawDDL` | 110 | **70.9** | 81.1 | 73.0 | 58.3 | 71.6 | 5238.2 | 1926 |
| BIRD-Dev Clean | `B0+_EnrichedMDDL` | 110 | **72.7** | 81.1 | 75.7 | 61.1 | 73.0 | 5869.0 | 3647 |
| BIRD-Dev Clean | `B1_LexicalCoT` | 110 | **72.7** | 83.8 | 75.7 | 58.3 | 73.6 | 6329.8 | 3632 |
| BIRD-Dev Clean | `B2_ExecRefiner` | 110 | **73.6** | 83.8 | 81.1 | 55.6 | 73.6 | 5730.7 | 2889 |
| BIRD-Dev Clean | `B3_ValueMajorityVote` | 110 | **77.3** | 81.1 | 86.5 | 63.9 | 78.0 | 6207.4 | 11416 |
| BIRD-Dev Clean | `B4_LLMJudge` | 110 | **76.4** | 83.8 | 86.5 | 58.3 | 77.0 | 8125.7 | 11783 |
| BIRD-Dev Clean | `Argus-SQL` | 110 | **75.5** | 86.5 | 81.1 | 58.3 | 76.8 | 11066.6 | 11914 |
| BIRD-Dev Voice | `B0_RawDDL` | 110 | **69.1** | 78.4 | 73.0 | 55.6 | 69.5 | 5793.3 | 1944 |
| BIRD-Dev Voice | `B0+_EnrichedMDDL` | 110 | **73.6** | 81.1 | 78.4 | 61.1 | 73.9 | 6599.6 | 3650 |
| BIRD-Dev Voice | `B1_LexicalCoT` | 110 | **70.9** | 78.4 | 75.7 | 58.3 | 70.9 | 6763.9 | 3622 |
| BIRD-Dev Voice | `B2_ExecRefiner` | 110 | **70.9** | 78.4 | 78.4 | 55.6 | 70.0 | 4234.7 | 2307 |
| BIRD-Dev Voice | `B3_ValueMajorityVote` | 110 | **74.5** | 83.8 | 81.1 | 58.3 | 75.0 | 7259.7 | 11514 |
| BIRD-Dev Voice | `B4_LLMJudge` | 110 | **72.7** | 81.1 | 78.4 | 58.3 | 73.2 | 9549.6 | 11945 |
| BIRD-Dev Voice | `Argus-SQL` | 110 | **77.3** | 86.5 | 83.8 | 61.1 | 78.0 | 360.3 | 11898 |

- **Paired McNemar (`Argus-SQL` vs `B0_RawDDL`, Clean, N=110)**: `p = 2.266e-01`

- **Paired McNemar (`Argus-SQL` vs `B0+_EnrichedMDDL`, Clean, N=110)**: `p = 4.531e-01`

- **Paired McNemar (`Argus-SQL` vs `B1_LexicalCoT`, Clean, N=110)**: `p = 3.750e-01`

- **Paired McNemar (`Argus-SQL` vs `B2_ExecRefiner`, Clean, N=110)**: `p = 6.875e-01`

- **Paired McNemar (`Argus-SQL` vs `B3_ValueMajorityVote`, Clean, N=110)**: `p = 6.875e-01`

- **Paired McNemar (`Argus-SQL` vs `B4_LLMJudge`, Clean, N=110)**: `p = 1.000e+00`

## 3. Per-Database Accuracy (`Argus-SQL` Clean Split)

| Database (`db_id`) | Tasks | EX (%) |
| :--- | :---: | :---: |
| `card_games` | 10 | 50.0% |
| `california_schools` | 10 | 60.0% |
| `codebase_community` | 10 | 70.0% |
| `financial` | 10 | 70.0% |
| `toxicology` | 10 | 70.0% |
| `debit_card_specializing` | 10 | 80.0% |
| `european_football_2` | 10 | 80.0% |
| `formula_1` | 10 | 80.0% |
| `thrombosis_prediction` | 10 | 80.0% |
| `student_club` | 10 | 90.0% |
| `superhero` | 10 | 100.0% |

## 4. Ranked Failure Clusters (`Argus-SQL`)

| Count | Category | Sub-Signature | Databases | Sample QIDs | Recommended Module Action |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 24 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `UNFLAGGED_LOGIC_OR_AGGREGATION_DRIFT` | `california_schools`, `card_games`, `codebase_community`, `debit_card_specializing` | `344`, `586`, `94`, `847`, `197`, `198` | Compare `predicted_sql` against domain formulas; add new SQLite invariant. |
| 2 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:LEFT_JOIN_FILTERED_RATIO` | `card_games` | `371`, `416` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:EXECUTION_ERROR` | `debit_card_specializing` | `1482` | Strengthen consensus penalty or algebraic repair in `verification/`. |

## 5. Spoken Voice-BIRD Drift Analysis

- **QID `24` (`california_schools`, `moderate`)**: Clean EX=`False` -> Voice EX=`True` (`CLEAN_MATCH`)
- **QID `416` (`card_games`, `challenging`)**: Clean EX=`False` -> Voice EX=`True` (`CLEAN_MATCH`)

## 6. Sample Sanitized Failed SQL Traces

### QID `344` (`card_games` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot+argus_repair` | **Invariants**: `none`
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
  T1.DisplayName
FROM users AS T1
JOIN votes AS T2
  ON T1.Id = T2.UserId
JOIN posts AS T3
  ON T2.PostId = T3.Id
WHERE
  T2.BountyAmount = 50
  AND T3.Title LIKE '%variance%'
```

### QID `94` (`financial` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  T1.account_id,
  (SELECT MAX(A11) FROM district) - T3.A11
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
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  T2.surname
FROM qualifying AS T1
INNER JOIN drivers AS T2
  ON T1.driverId = T2.driverId
WHERE T1.raceId = 19
  AND T1.q2 IS NOT NULL
ORDER BY T1.q2 ASC
LIMIT 1
```

### QID `197` (`toxicology` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT
  AVG(oxygen_count)
FROM (
  SELECT
    COUNT(CASE WHEN element = 'o' THEN 1 END) AS oxygen_count
  FROM atom
  WHERE molecule_id IN (
    SELECT molecule_id
    FROM bond
    WHERE bond_type = '-'
  )
  GROUP BY molecule_id
)
```

### QID `371` (`card_games` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `LEFT_JOIN_FILTERED_RATIO`
```sql
SELECT
  100.0 * COUNT(DISTINCT CASE WHEN T2.language = 'French' THEN T1.id END) / COUNT(DISTINCT T1.id)
FROM cards AS T1
LEFT JOIN foreign_data AS T2
  ON T1.uuid = T2.uuid
WHERE T1.isStorySpotlight = 1
```

### QID `198` (`toxicology` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT
  CAST(SUM(CASE WHEN T2.bond_type = '-' THEN 1 ELSE 0 END) AS REAL) / COUNT(T3.atom_id)
FROM molecule AS T1
INNER JOIN bond AS T2
  ON T1.molecule_id = T2.molecule_id
INNER JOIN atom AS T3
  ON T1.molecule_id = T3.molecule_id
WHERE T1.label = '+'
```

### QID `28` (`california_schools` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
```sql
SELECT
  T1.School,
  T1.DOCType
FROM schools AS T1
INNER JOIN frpm AS T2
  ON T1.CDSCode = T2.CDSCode
WHERE (T2."Enrollment (K-12)" - T2."Enrollment (Ages 5-17)") > (
  SELECT AVG("Enrollment (K-12)" - "Enrollment (Ages 5-17)")
  FROM frpm
  WHERE "Charter Funding Type" = 'Locally funded'
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
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal+argus_repair` | **Invariants**: `none`
```sql
SELECT
  T2.Date,
  strftime('%Y', T1."First Date") - strftime('%Y', T1.Birthday)
FROM Patient AS T1
JOIN Laboratory AS T2
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
GROUP BY team_api_id
ORDER BY MAX(buildUpPlaySpeed) DESC
LIMIT 4
```

### QID `115` (`financial` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  CAST(SUM(CASE WHEN gender = 'M' THEN 1 ELSE 0 END) AS REAL) * 100.0 / NULLIF(COUNT(client_id), 0)
FROM client
WHERE district_id = (
  SELECT district_id
  FROM district
  WHERE A3 = 'south Bohemia'
  ORDER BY CAST(A4 AS INTEGER) DESC
  LIMIT 1
)
```
