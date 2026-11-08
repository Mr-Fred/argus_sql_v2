# Argus-SQL Split-Environment Feedback Triage Report

## 1. Run Provenance & Integrity

- **Schema**: `argus_feedback.v1`
- **Created (UTC)**: `2026-11-08T15:03:16Z`
- **Provider / Model**: `vertex` / `gemini-3.7-flash`
- **Dataset Split**: `mini_dev_sqlite.json`
- **Source Commit**: `b34c0ab16009`
- **Payload SHA-256**: `9ba1fe2e3eec8a8f...` (verified)

## 2. Leaderboard Summary

| Suite | System | Tasks | EX (%) | Simple | Mod. | Chall. | R-VES | p50 (ms) | Mean Tok |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BIRD-Dev Clean | `Argus-SQL` | 20 | **60.0** | 66.7 | 75.0 | 0.0 | 52.5 | 7999.0 | 4054 |
| BIRD-Dev Voice | `Argus-SQL` | 20 | **65.0** | 77.8 | 75.0 | 0.0 | 57.5 | 8209.1 | 3665 |

## 3. Per-Database Accuracy (`Argus-SQL` Clean Split)

| Database (`db_id`) | Tasks | EX (%) |
| :--- | :---: | :---: |
| `debit_card_specializing` | 20 | 60.0% |

## 4. Ranked Failure Clusters (`Argus-SQL`)

| Count | Category | Sub-Signature | Databases | Sample QIDs | Recommended Module Action |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 4 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `UNFLAGGED_LOGIC_OR_AGGREGATION_DRIFT` | `debit_card_specializing` | `1476`, `1498`, `1505`, `1506` | Compare `predicted_sql` against domain formulas; add new SQLite invariant. |
| 1 | `EMPTY_RESULT_VALUE_DRIFT` | `EMPTY_PREDICTED_RESULT` | `debit_card_specializing` | `1500` | Expand 3-gram cell literal top-k or case-insensitive match in `linking/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:EXECUTION_ERROR` | `debit_card_specializing` | `1490` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:EXECUTION_ERROR,INTEGER_DIVISION_TRUNCATION` | `debit_card_specializing` | `1482` | Strengthen consensus penalty or algebraic repair in `verification/`. |
| 1 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `INVARIANT_FLAGGED:PROJECTION_ARITY_DRIFT` | `debit_card_specializing` | `1481` | Strengthen consensus penalty or algebraic repair in `verification/`. |

## 5. Spoken Voice-BIRD Drift Analysis

- **QID `1480` (`debit_card_specializing`, `moderate`)**: Clean EX=`True` -> Voice EX=`False` (`SILENT_SEMANTIC_OR_FORMULA_ERROR`)
- **QID `1498` (`debit_card_specializing`, `simple`)**: Clean EX=`False` -> Voice EX=`True` (`CLEAN_MATCH`)
- **QID `1506` (`debit_card_specializing`, `moderate`)**: Clean EX=`False` -> Voice EX=`True` (`CLEAN_MATCH`)

## 6. Sample Sanitized Failed SQL Traces

### QID `1476` (`debit_card_specializing` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
```sql
SELECT
  SUM(CASE WHEN T2.Currency = 'CZK' THEN T1.Consumption ELSE 0 END) - SUM(CASE WHEN T2.Currency = 'EUR' THEN T1.Consumption ELSE 0 END)
FROM yearmonth AS T1
INNER JOIN customers AS T2
  ON T1.CustomerID = T2.CustomerID
WHERE
  T1.Date BETWEEN 201201 AND 201212
```

### QID `1481` (`debit_card_specializing` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `PROJECTION_ARITY_DRIFT`
```sql
WITH customer_consumption AS (
    SELECT
        T2.Segment,
        T1.CustomerID,
        SUM(T1.Consumption) AS TotalConsumption
    FROM yearmonth AS T1
    JOIN customers AS T2 ON T1.CustomerID = T2.CustomerID
    WHERE T1.Date BETWEEN '201301' AND '201312'
      AND T2.Currency = 'CZK'
    GROUP BY T2.Segment, T1.CustomerID
),
min_segment_consumption AS (
    SELECT
        Segment,
        MIN(TotalConsumption) AS MinConsumption
    FROM customer_consumption
    GROUP BY Segment
)
SELECT
    (SELECT MinConsumption FROM min_segment_consumption WHERE Segment = 'SME') - 
    (SELECT MinConsumption FROM min_segment_consumption WHERE Segment = 'LAM') AS diff_SME_LAM,
    (SELECT MinConsumption FROM min_segment_consumption WHERE Segment = 'LAM') - 
    (SELECT MinConsumption FROM min_segment_consumption WHERE Segment = 'KAM') AS diff_LAM_KAM,
    (SELECT MinConsumption FROM min_segment_consumption WHERE Segment = 'KAM') - 
    (SELECT MinConsumption FROM min_segment_consumption WHERE Segment = 'SME') AS diff_KAM_SME
```

### QID `1482` (`debit_card_specializing` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `EXECUTION_ERROR,INTEGER_DIVISION_TRUNCATION`
```sql
WITH segment_consumption AS (
  SELECT
    T1.Segment,
    SUM(CASE WHEN SUBSTR(T2.Date, 1, 4) = '2012' THEN T2.Consumption ELSE 0 END) AS consumption_2012,
    SUM(CASE WHEN SUBSTR(T2.Date, 1, 4) = '2013' THEN T2.Consumption ELSE 0 END) AS consumption_2013
  FROM customers AS T1
  INNER JOIN yearmonth AS T2
    ON T1.CustomerID = T2.CustomerID
  WHERE T1.Currency = 'EUR'
    AND T1.Segment IN ('SME', 'LAM', 'KAM')
  GROUP BY T1.Segment
),
percentage_increase AS (
  SELECT
    Segment,
    CAST(consumption_2013 - consumption_2012 AS REAL) / NULLIF(consumption_2013, 0) * 100 AS pct_increase
  FROM segment_consumption
)
SELECT Segment
FROM (
  SELECT Segment, pct_increase FROM percentage_increase WHERE pct_increase IS NOT NULL ORDER BY pct_increase DESC LIMIT 1
)
UNION ALL
SELECT Segment
FROM (
  SELECT Segment, pct_increase FROM percentage_increase WHERE pct_increase IS NOT NULL ORDER BY pct_increase ASC LIMIT 1
)
```

### QID `1490` (`debit_card_specializing` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `argus_llm_repair` | **Invariants**: `EXECUTION_ERROR`
```sql
SELECT
  100.0 * COUNT(CASE WHEN total_consumption > 46.73 THEN 1 END) / COUNT(*)
FROM (
  SELECT
    T1.CustomerID,
    SUM(T2.Consumption) AS total_consumption
  FROM customers AS T1
  LEFT JOIN yearmonth AS T2
    ON T1.CustomerID = T2.CustomerID
  WHERE T1.Segment = 'LAM'
  GROUP BY T1.CustomerID
)
```

### QID `1498` (`debit_card_specializing` — `simple`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT
  MAX(Consumption)
FROM yearmonth
WHERE Date LIKE '2012%'
```

### QID `1500` (`debit_card_specializing` — `simple`)
- **Category**: `EMPTY_RESULT_VALUE_DRIFT` | **Strategy**: `argus_llm_repair` | **Invariants**: `EMPTY_RESULT_ANOMALY,EXECUTION_ERROR`
```sql
SELECT DISTINCT T1.Description
FROM products AS T1
JOIN transactions_1k AS T2
  ON T1.ProductID = T2.ProductID
WHERE strftime('%Y%m', T2.Date) = '201309'
   OR T2.Date LIKE '2013-09%'
```

### QID `1505` (`debit_card_specializing` — `simple`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT count(DISTINCT T1.CustomerID)
FROM customers AS T1
INNER JOIN yearmonth AS T2
  ON T1.CustomerID = T2.CustomerID
WHERE T1.Currency = 'EUR'
  AND T2.Consumption > 1000
```

### QID `1506` (`debit_card_specializing` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT
  T1.Description
FROM products AS T1
JOIN transactions_1k AS T2
  ON T1.ProductID = T2.ProductID
JOIN gasstations AS T3
  ON T2.GasStationID = T3.GasStationID
WHERE
  T3.Country = 'CZE'
```
