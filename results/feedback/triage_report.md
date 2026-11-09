# Argus-SQL Split-Environment Feedback Triage Report

## 1. Run Provenance & Integrity

- **Schema**: `argus_feedback.v1`
- **Created (UTC)**: `2026-11-09T07:02:21Z`
- **Provider / Model**: `vertex` / `gemini-3.7-flash`
- **Dataset Split**: `mini_dev_sqlite.json`
- **Source Commit**: `3684d41e24a0`
- **Payload SHA-256**: `767d7ed5f22ea1b1...` (verified)

## 2. Leaderboard Summary

| Suite | System | Tasks | EX (%) | Simple | Mod. | Chall. | R-VES | p50 (ms) | Mean Tok |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BIRD-Dev Clean | `B0_ZeroShotDDL` | 22 | **68.2** | 85.7 | 62.5 | 57.1 | 67.0 | 5532.2 | 3582 |
| BIRD-Dev Clean | `B1_DINDecomposed` | 22 | **59.1** | 85.7 | 62.5 | 28.6 | 59.1 | 5355.9 | 3461 |
| BIRD-Dev Clean | `B2_MACRefiner` | 22 | **63.6** | 85.7 | 75.0 | 28.6 | 59.1 | 56.0 | 236 |
| BIRD-Dev Clean | `B3_CHESSValueConsistency` | 22 | **59.1** | 85.7 | 62.5 | 28.6 | 58.0 | 16037.4 | 10355 |
| BIRD-Dev Clean | `Argus-SQL` | 22 | **72.7** | 85.7 | 75.0 | 57.1 | 71.6 | 10109.4 | 10915 |
| BIRD-Dev Voice | `B0_ZeroShotDDL` | 22 | **72.7** | 85.7 | 75.0 | 57.1 | 69.3 | 6626.4 | 3523 |
| BIRD-Dev Voice | `B1_DINDecomposed` | 22 | **59.1** | 85.7 | 62.5 | 28.6 | 59.1 | 6098.2 | 3460 |
| BIRD-Dev Voice | `B2_MACRefiner` | 22 | **63.6** | 85.7 | 75.0 | 28.6 | 56.8 | 50.6 | 133 |
| BIRD-Dev Voice | `B3_CHESSValueConsistency` | 22 | **63.6** | 85.7 | 75.0 | 28.6 | 62.5 | 14379.3 | 10373 |
| BIRD-Dev Voice | `Argus-SQL` | 22 | **72.7** | 85.7 | 75.0 | 57.1 | 70.5 | 7429.6 | 10763 |

- **Paired McNemar (`Argus-SQL` vs `B0_ZeroShotDDL`, Clean, N=22)**: `p = 1.000e+00`

- **Paired McNemar (`Argus-SQL` vs `B1_DINDecomposed`, Clean, N=22)**: `p = 2.500e-01`

- **Paired McNemar (`Argus-SQL` vs `B2_MACRefiner`, Clean, N=22)**: `p = 5.000e-01`

- **Paired McNemar (`Argus-SQL` vs `B3_CHESSValueConsistency`, Clean, N=22)**: `p = 2.500e-01`

## 3. Per-Database Accuracy (`Argus-SQL` Clean Split)

| Database (`db_id`) | Tasks | EX (%) |
| :--- | :---: | :---: |
| `toxicology` | 2 | 0.0% |
| `card_games` | 2 | 50.0% |
| `codebase_community` | 2 | 50.0% |
| `financial` | 2 | 50.0% |
| `formula_1` | 2 | 50.0% |
| `california_schools` | 2 | 100.0% |
| `debit_card_specializing` | 2 | 100.0% |
| `european_football_2` | 2 | 100.0% |
| `student_club` | 2 | 100.0% |
| `superhero` | 2 | 100.0% |
| `thrombosis_prediction` | 2 | 100.0% |

## 4. Ranked Failure Clusters (`Argus-SQL`)

| Count | Category | Sub-Signature | Databases | Sample QIDs | Recommended Module Action |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 5 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `UNFLAGGED_LOGIC_OR_AGGREGATION_DRIFT` | `codebase_community`, `financial`, `formula_1`, `toxicology` | `586`, `94`, `847`, `197`, `198` | Compare `predicted_sql` against domain formulas; add new SQLite invariant. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:JOIN_FANOUT_DUPLICATES` | `card_games` | `344` | Strengthen consensus penalty or algebraic repair in `verification/`. |

## 5. Spoken Voice-BIRD Drift Analysis

Zero Clean-vs-Voice discrepancies observed.

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
