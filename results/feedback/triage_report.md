# Argus-SQL Split-Environment Feedback Triage Report

## 1. Run Provenance & Integrity

- **Schema**: `argus_feedback.v1`
- **Created (UTC)**: `2026-11-09T08:08:35Z`
- **Provider / Model**: `vertex` / `gemini-3.7-flash`
- **Dataset Split**: `mini_dev_sqlite.json`
- **Source Commit**: `bfb040275434`
- **Payload SHA-256**: `f4419f770134b474...` (verified)

## 2. Leaderboard Summary

| Suite | System | Tasks | EX (%) | Simple | Mod. | Chall. | R-VES | p50 (ms) | Mean Tok |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BIRD-Dev Clean | `B0_ZeroShotDDL` | 110 | **72.7** | 81.1 | 75.7 | 61.1 | 73.0 | 5869.0 | 3647 |
| BIRD-Dev Clean | `B1_DINDecomposed` | 110 | **72.7** | 83.8 | 75.7 | 58.3 | 73.6 | 6329.8 | 3632 |
| BIRD-Dev Clean | `B2_MACRefiner` | 110 | **73.6** | 83.8 | 81.1 | 55.6 | 73.6 | 5730.7 | 2889 |
| BIRD-Dev Clean | `B3_CHESSValueConsistency` | 110 | **70.9** | 78.4 | 78.4 | 55.6 | 71.6 | 11563.2 | 10878 |
| BIRD-Dev Clean | `Argus-SQL` | 110 | **74.5** | 81.1 | 78.4 | 63.9 | 75.2 | 9921.2 | 10985 |
| BIRD-Dev Voice | `B0_ZeroShotDDL` | 110 | **73.6** | 81.1 | 78.4 | 61.1 | 73.9 | 6599.6 | 3650 |
| BIRD-Dev Voice | `B1_DINDecomposed` | 110 | **70.9** | 78.4 | 75.7 | 58.3 | 70.9 | 6763.9 | 3622 |
| BIRD-Dev Voice | `B2_MACRefiner` | 110 | **70.9** | 78.4 | 78.4 | 55.6 | 70.0 | 4234.7 | 2307 |
| BIRD-Dev Voice | `B3_CHESSValueConsistency` | 110 | **70.9** | 78.4 | 78.4 | 55.6 | 71.4 | 12642.5 | 10933 |
| BIRD-Dev Voice | `Argus-SQL` | 110 | **71.8** | 81.1 | 78.4 | 55.6 | 72.3 | 11395.7 | 11145 |

- **Paired McNemar (`Argus-SQL` vs `B0_ZeroShotDDL`, Clean, N=110)**: `p = 6.875e-01`

- **Paired McNemar (`Argus-SQL` vs `B1_DINDecomposed`, Clean, N=110)**: `p = 6.875e-01`

- **Paired McNemar (`Argus-SQL` vs `B2_MACRefiner`, Clean, N=110)**: `p = 1.000e+00`

- **Paired McNemar (`Argus-SQL` vs `B3_CHESSValueConsistency`, Clean, N=110)**: `p = 2.188e-01`

## 3. Per-Database Accuracy (`Argus-SQL` Clean Split)

| Database (`db_id`) | Tasks | EX (%) |
| :--- | :---: | :---: |
| `california_schools` | 10 | 50.0% |
| `card_games` | 10 | 70.0% |
| `codebase_community` | 10 | 70.0% |
| `financial` | 10 | 70.0% |
| `formula_1` | 10 | 70.0% |
| `toxicology` | 10 | 70.0% |
| `debit_card_specializing` | 10 | 80.0% |
| `european_football_2` | 10 | 80.0% |
| `thrombosis_prediction` | 10 | 80.0% |
| `student_club` | 10 | 90.0% |
| `superhero` | 10 | 90.0% |

## 4. Ranked Failure Clusters (`Argus-SQL`)

| Count | Category | Sub-Signature | Databases | Sample QIDs | Recommended Module Action |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 23 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `UNFLAGGED_LOGIC_OR_AGGREGATION_DRIFT` | `california_schools`, `card_games`, `codebase_community`, `debit_card_specializing` | `586`, `94`, `847`, `197`, `198`, `28` | Compare `predicted_sql` against domain formulas; add new SQLite invariant. |
| 2 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:PROJECTION_ARITY_DRIFT` | `california_schools`, `formula_1` | `36`, `866` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:EXECUTION_ERROR` | `debit_card_specializing` | `1482` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:JOIN_FANOUT_DUPLICATES` | `card_games` | `344` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:UNFILTERED_NULL_ASC_LIMIT` | `thrombosis_prediction` | `1168` | Strengthen consensus penalty or algebraic repair in `verification/`. |

## 5. Spoken Voice-BIRD Drift Analysis

- **QID `206` (`toxicology`, `challenging`)**: Clean EX=`True` -> Voice EX=`False` (`SILENT_SEMANTIC_OR_FORMULA_ERROR`)
- **QID `743` (`superhero`, `challenging`)**: Clean EX=`True` -> Voice EX=`False` (`SILENT_SEMANTIC_OR_FORMULA_ERROR`)
- **QID `744` (`superhero`, `challenging`)**: Clean EX=`True` -> Voice EX=`False` (`SILENT_SEMANTIC_OR_FORMULA_ERROR`)

## 6. Sample Sanitized Failed SQL Traces

### QID `344` (`card_games` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal+argus_repair` | **Invariants**: `JOIN_FANOUT_DUPLICATES`
```sql
SELECT DISTINCT T1.name
FROM cards AS T1
INNER JOIN legalities AS T2
  ON T1.uuid = T2.uuid
WHERE T1.rarity = 'mythic'
  AND T2.status = 'Banned'
  AND T2.format = 'gladiator'
```

### QID `586` (`codebase_community` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT DISTINCT
  T1.DisplayName
FROM users AS T1
INNER JOIN votes AS T2
  ON T1.Id = T2.UserId
INNER JOIN posts AS T3
  ON T2.PostId = T3.Id
WHERE T2.BountyAmount = 50
  AND T3.Title LIKE '%variance%'
```

### QID `94` (`financial` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
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
  T1.surname
FROM drivers AS T1
INNER JOIN qualifying AS T2
  ON T1.driverId = T2.driverId
WHERE
  T2.raceId = 19
  AND T2.q2 IS NOT NULL
  AND T2.q2 != ''
ORDER BY
  T2.q2 ASC
LIMIT 1
```

### QID `197` (`toxicology` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT AVG(cnt)
FROM (
  SELECT COUNT(atom_id) AS cnt
  FROM atom
  WHERE element = 'o'
    AND molecule_id IN (
      SELECT molecule_id
      FROM bond
      WHERE bond_type = '-'
    )
  GROUP BY molecule_id
)
```

### QID `198` (`toxicology` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
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
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  T1.School,
  T1.DOCType
FROM schools AS T1
INNER JOIN frpm AS T2
  ON T1.CDSCode = T2.CDSCode
WHERE (T2."Enrollment (K-12)" - T2."Enrollment (Ages 5-17)") > (
  SELECT AVG(T3."Enrollment (K-12)" - T3."Enrollment (Ages 5-17)")
  FROM frpm AS T3
  WHERE T3."Charter Funding Type" = 'Locally funded'
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
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `UNFILTERED_NULL_ASC_LIMIT`
```sql
SELECT
  T2.Date,
  CAST(strftime('%Y', T1."First Date") AS INTEGER) - CAST(strftime('%Y', T1.Birthday) AS INTEGER) AS age_at_arrival
FROM Patient AS T1
INNER JOIN Laboratory AS T2
  ON T1.ID = T2.ID
WHERE T1.Diagnosis = 'SJS'
  AND T1.Birthday IS NOT NULL
ORDER BY T1.Birthday ASC
LIMIT 1
```

### QID `1029` (`european_football_2` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  MAX(buildUpPlaySpeed)
FROM Team_Attributes
WHERE
  buildUpPlaySpeed IS NOT NULL
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
  CAST(SUM(CASE WHEN T2.gender = 'M' THEN 1 ELSE 0 END) AS REAL) * 100.0 / COUNT(T2.client_id)
FROM district AS T1
INNER JOIN client AS T2
  ON T1.district_id = T2.district_id
WHERE T1.district_id = (
  SELECT district_id
  FROM district
  WHERE A3 = 'south Bohemia'
  ORDER BY CAST(A4 AS INTEGER) DESC
  LIMIT 1
)
```

### QID `1322` (`student_club` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  COUNT(*)
FROM (
  SELECT
    T1.event_id
  FROM event AS T1
  INNER JOIN attendance AS T2
    ON T1.event_id = T2.link_to_event
  WHERE T1.type = 'Meeting'
  GROUP BY T1.event_id
  HAVING COUNT(T2.link_to_member) > 10
)
```
