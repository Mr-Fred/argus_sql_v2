# Argus-SQL Split-Environment Feedback Triage Report

## 1. Run Provenance & Integrity

- **Schema**: `argus_feedback.v1`
- **Created (UTC)**: `2026-11-09T02:31:39Z`
- **Provider / Model**: `vertex` / `gemini-3.7-flash`
- **Dataset Split**: `mini_dev_sqlite.json`
- **Source Commit**: `e9079292b150`
- **Payload SHA-256**: `bef9ca3a6291ab70...` (verified)

## 2. Leaderboard Summary

| Suite | System | Tasks | EX (%) | Simple | Mod. | Chall. | R-VES | p50 (ms) | Mean Tok |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BIRD-Dev Clean | `B0_ZeroShotDDL` | 22 | **68.2** | 85.7 | 62.5 | 57.1 | 67.0 | 5532.2 | 3582 |
| BIRD-Dev Clean | `B1_DINDecomposed` | 22 | **59.1** | 85.7 | 62.5 | 28.6 | 59.1 | 5355.9 | 3461 |
| BIRD-Dev Clean | `B2_MACRefiner` | 22 | **63.6** | 85.7 | 75.0 | 28.6 | 59.1 | 56.0 | 236 |
| BIRD-Dev Clean | `B3_CHESSValueConsistency` | 22 | **59.1** | 85.7 | 62.5 | 28.6 | 58.0 | 16037.4 | 10355 |
| BIRD-Dev Clean | `Argus-SQL` | 22 | **63.6** | 85.7 | 75.0 | 28.6 | 62.5 | 18334.3 | 10398 |
| BIRD-Dev Voice | `B0_ZeroShotDDL` | 22 | **72.7** | 85.7 | 75.0 | 57.1 | 69.3 | 6626.4 | 3523 |
| BIRD-Dev Voice | `B1_DINDecomposed` | 22 | **59.1** | 85.7 | 62.5 | 28.6 | 59.1 | 6098.2 | 3460 |
| BIRD-Dev Voice | `B2_MACRefiner` | 22 | **63.6** | 85.7 | 75.0 | 28.6 | 56.8 | 50.6 | 133 |
| BIRD-Dev Voice | `B3_CHESSValueConsistency` | 22 | **63.6** | 85.7 | 75.0 | 28.6 | 62.5 | 14379.3 | 10373 |
| BIRD-Dev Voice | `Argus-SQL` | 22 | **63.6** | 85.7 | 75.0 | 28.6 | 62.5 | 11452.9 | 10316 |

- **Paired McNemar (`Argus-SQL` vs `B0_ZeroShotDDL`, Clean, N=22)**: `p = 1.000e+00`

- **Paired McNemar (`Argus-SQL` vs `B1_DINDecomposed`, Clean, N=22)**: `p = 1.000e+00`

- **Paired McNemar (`Argus-SQL` vs `B2_MACRefiner`, Clean, N=22)**: `p = 1.000e+00`

- **Paired McNemar (`Argus-SQL` vs `B3_CHESSValueConsistency`, Clean, N=22)**: `p = 1.000e+00`

## 3. Per-Database Accuracy (`Argus-SQL` Clean Split)

| Database (`db_id`) | Tasks | EX (%) |
| :--- | :---: | :---: |
| `card_games` | 2 | 0.0% |
| `toxicology` | 2 | 0.0% |
| `codebase_community` | 2 | 50.0% |
| `european_football_2` | 2 | 50.0% |
| `financial` | 2 | 50.0% |
| `formula_1` | 2 | 50.0% |
| `california_schools` | 2 | 100.0% |
| `debit_card_specializing` | 2 | 100.0% |
| `student_club` | 2 | 100.0% |
| `superhero` | 2 | 100.0% |
| `thrombosis_prediction` | 2 | 100.0% |

## 4. Ranked Failure Clusters (`Argus-SQL`)

| Count | Category | Sub-Signature | Databases | Sample QIDs | Recommended Module Action |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 7 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `UNFLAGGED_LOGIC_OR_AGGREGATION_DRIFT` | `card_games`, `codebase_community`, `european_football_2`, `financial` | `586`, `94`, `847`, `197`, `371`, `1028` | Compare `predicted_sql` against domain formulas; add new SQLite invariant. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:JOIN_FANOUT_DUPLICATES` | `card_games` | `344` | Strengthen consensus penalty or algebraic repair in `verification/`. |

## 5. Spoken Voice-BIRD Drift Analysis

Zero Clean-vs-Voice discrepancies observed.

## 6. Sample Sanitized Failed SQL Traces

### QID `344` (`card_games` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot+argus_repair` | **Invariants**: `JOIN_FANOUT_DUPLICATES`
```sql
SELECT DISTINCT T1.name
FROM cards AS T1
INNER JOIN legalities AS T2
  ON T1.uuid = T2.uuid
WHERE
  T1.rarity = 'mythic'
  AND T2.format = 'gladiator'
  AND T2.status = 'Banned'
```

### QID `586` (`codebase_community` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
```sql
SELECT
  T1.DisplayName
FROM users AS T1
INNER JOIN votes AS T2
  ON T1.Id = T2.UserId
INNER JOIN posts AS T3
  ON T2.PostId = T3.Id
WHERE
  T2.BountyAmount = 50
  AND T3.Title LIKE '%variance%'
```

### QID `94` (`financial` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
```sql
SELECT
  T3.account_id,
  (SELECT MAX(A11) FROM district) - T2.A11
FROM client AS T1
JOIN district AS T2
  ON T1.district_id = T2.district_id
JOIN disp AS T3
  ON T1.client_id = T3.client_id
WHERE T1.gender = 'F'
ORDER BY T1.birth_date ASC, T2.A11 ASC
LIMIT 1
```

### QID `847` (`formula_1` — `simple`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT T2.surname
FROM qualifying AS T1
INNER JOIN drivers AS T2
  ON T1.driverId = T2.driverId
WHERE T1.raceId = 19
  AND T1.q2 IS NOT NULL
ORDER BY T1.q2 ASC
LIMIT 1
```

### QID `197` (`toxicology` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  AVG(T1.element = 'o')
FROM atom AS T1
INNER JOIN bond AS T2
  ON T1.molecule_id = T2.molecule_id
WHERE T2.bond_type = '-'
```

### QID `371` (`card_games` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  CAST(COUNT(T2.id) AS REAL) * 100.0 / NULLIF(COUNT(T1.id), 0)
FROM cards AS T1
LEFT JOIN foreign_data AS T2
  ON T1.uuid = T2.uuid AND T2.language = 'French'
WHERE T1.isStorySpotlight = 1
```

### QID `1028` (`european_football_2` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  T.team_long_name
FROM Match AS M
INNER JOIN League AS L
  ON M.league_id = L.id
INNER JOIN Team AS T
  ON M.away_team_api_id = T.team_api_id
WHERE L.name = 'Scotland Premier League'
  AND M.season = '2009/2010'
  AND M.away_team_goal > M.home_team_goal
GROUP BY
  T.team_long_name
ORDER BY
  COUNT(*) DESC
LIMIT 1
```

### QID `198` (`toxicology` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT
  CAST(SUM(CASE WHEN T1.bond_type = '-' THEN 1 ELSE 0 END) AS REAL) / COUNT(T2.atom_id)
FROM bond AS T1
INNER JOIN molecule AS T3
  ON T1.molecule_id = T3.molecule_id
INNER JOIN atom AS T2
  ON T3.molecule_id = T2.molecule_id
WHERE T3.label = '+'
```
