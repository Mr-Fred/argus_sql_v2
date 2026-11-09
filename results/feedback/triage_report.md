# Argus-SQL Split-Environment Feedback Triage Report

## 1. Run Provenance & Integrity

- **Schema**: `argus_feedback.v1`
- **Created (UTC)**: `2026-11-08T23:48:46Z`
- **Provider / Model**: `vertex` / `gemini-3.7-flash`
- **Dataset Split**: `mini_dev_sqlite.json`
- **Source Commit**: `085a85eefce0`
- **Payload SHA-256**: `9912a8611aff7172...` (verified)

## 2. Leaderboard Summary

| Suite | System | Tasks | EX (%) | Simple | Mod. | Chall. | R-VES | p50 (ms) | Mean Tok |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BIRD-Dev Clean | `Argus-SQL` | 20 | **65.0** | 66.7 | 75.0 | 33.3 | 61.2 | 10868.9 | 5020 |
| BIRD-Dev Voice | `Argus-SQL` | 20 | **65.0** | 66.7 | 75.0 | 33.3 | 58.8 | 8042.5 | 4875 |

## 3. Per-Database Accuracy (`Argus-SQL` Clean Split)

| Database (`db_id`) | Tasks | EX (%) |
| :--- | :---: | :---: |
| `debit_card_specializing` | 20 | 65.0% |

## 4. Ranked Failure Clusters (`Argus-SQL`)

| Count | Category | Sub-Signature | Databases | Sample QIDs | Recommended Module Action |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 5 | `SILENT_SEMANTIC_OR_FORMULA_ERROR` | `UNFLAGGED_LOGIC_OR_AGGREGATION_DRIFT` | `debit_card_specializing` | `1481`, `1482`, `1490`, `1498`, `1505` | Compare `predicted_sql` against domain formulas; add new SQLite invariant. |
| 2 | `EMPTY_RESULT_VALUE_DRIFT` | `EMPTY_PREDICTED_RESULT` | `debit_card_specializing` | `1500`, `1501` | Expand 3-gram cell literal top-k or case-insensitive match in `linking/`. |

## 5. Spoken Voice-BIRD Drift Analysis

Zero Clean-vs-Voice discrepancies observed.

## 6. Sample Sanitized Failed SQL Traces

### QID `1481` (`debit_card_specializing` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
WITH customer_consumption AS (
  SELECT
    T1.CustomerID,
    T2.Segment,
    SUM(T1.Consumption) AS total_consumption
  FROM yearmonth AS T1
  INNER JOIN customers AS T2
    ON T1.CustomerID = T2.CustomerID
  WHERE T1.Date BETWEEN '201301' AND '201312'
    AND T2.Currency = 'CZK'
  GROUP BY T1.CustomerID, T2.Segment
),
segment_min AS (
  SELECT
    Segment,
    MIN(total_consumption) AS min_consumption
  FROM customer_consumption
  GROUP BY Segment
)
SELECT
  (SELECT min_consumption FROM segment_min WHERE Segment = 'SME') - (SELECT min_consumption FROM segment_min WHERE Segment = 'LAM') AS diff_sme_lam,
  (SELECT min_consumption FROM segment_min WHERE Segment = 'LAM') - (SELECT min_consumption FROM segment_min WHERE Segment = 'KAM') AS diff_lam_kam,
  (SELECT min_consumption FROM segment_min WHERE Segment = 'KAM') - (SELECT min_consumption FROM segment_min WHERE Segment = 'SME') AS diff_kam_sme
```

### QID `1482` (`debit_card_specializing` — `challenging`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
```sql
SELECT
  T1.Segment,
  (SUM(CASE WHEN SUBSTR(T2.Date, 1, 4) = '2013' THEN T2.Consumption ELSE 0 END) - SUM(CASE WHEN SUBSTR(T2.Date, 1, 4) = '2012' THEN T2.Consumption ELSE 0 END)) * 100.0 / SUM(CASE WHEN SUBSTR(T2.Date, 1, 4) = '2013' THEN T2.Consumption ELSE 0 END) AS percentage_increase
FROM customers AS T1
INNER JOIN yearmonth AS T2
  ON T1.CustomerID = T2.CustomerID
WHERE T1.Currency = 'EUR'
  AND T1.Segment IN ('SME', 'LAM', 'KAM')
GROUP BY T1.Segment
ORDER BY percentage_increase DESC
```

### QID `1490` (`debit_card_specializing` — `moderate`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `decomposed_cot` | **Invariants**: `none`
```sql
SELECT
  CAST(COUNT(CASE WHEN T.total_consumption > 46.73 THEN 1 END) AS REAL) * 100 / COUNT(*)
FROM (
  SELECT
    T1.CustomerID,
    SUM(T2.Consumption) AS total_consumption
  FROM customers AS T1
  LEFT JOIN yearmonth AS T2
    ON T1.CustomerID = T2.CustomerID
  WHERE T1.Segment = 'LAM'
  GROUP BY T1.CustomerID
) AS T
```

### QID `1498` (`debit_card_specializing` — `simple`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `skeleton_minimal` | **Invariants**: `none`
```sql
SELECT
  MAX(Consumption)
FROM yearmonth
WHERE Date LIKE '2012%'
```

### QID `1500` (`debit_card_specializing` — `simple`)
- **Category**: `EMPTY_RESULT_VALUE_DRIFT` | **Strategy**: `formula_defensive` | **Invariants**: `EMPTY_RESULT_ANOMALY`
```sql
SELECT DISTINCT
  T1.Description
FROM products AS T1
JOIN transactions_1k AS T2
  ON T1.ProductID = T2.ProductID
WHERE T2.Date LIKE '2013-09%'
```

### QID `1501` (`debit_card_specializing` — `moderate`)
- **Category**: `EMPTY_RESULT_VALUE_DRIFT` | **Strategy**: `formula_defensive` | **Invariants**: `EMPTY_RESULT_ANOMALY`
```sql
SELECT DISTINCT
  T2.Country
FROM transactions_1k AS T1
JOIN gasstations AS T2
  ON T1.GasStationID = T2.GasStationID
WHERE T1.Date LIKE '2013-06%'
```

### QID `1505` (`debit_card_specializing` — `simple`)
- **Category**: `SILENT_SEMANTIC_OR_FORMULA_ERROR` | **Strategy**: `formula_defensive` | **Invariants**: `none`
```sql
SELECT
  COUNT(DISTINCT T1.CustomerID)
FROM customers AS T1
JOIN yearmonth AS T2
  ON T1.CustomerID = T2.CustomerID
WHERE T1.Currency = 'EUR'
  AND T2.Consumption > 1000
```
