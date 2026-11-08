"""Stratified Mini-BIRD and Voice-BIRD task specifications (48 queries total).

Defines 24 clean relational queries (`8 simple`, `8 moderate`, `8 challenging`)
and their 24 spoken `Voice-BIRD` counterparts spanning F1--F5 failure modes.
"""

from argus_sql.contracts.types import NLQueryTask
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.evaluation.bird_loader import perturb_task_for_voice

# Compact `(id, db_id, question, evidence, difficulty, gold_sql)` specs keep
# benchmark task definitions declarative, auditable, and concise.
_CLEAN_TASK_SPECS: tuple[tuple[str, str, str, str, QueryDifficulty, str], ...] = (
    # --- SIMPLE TIER (8 tasks: fast-path & orphan legacy-table pruning) ---
    ("mb_01", "schools_mini", "What is the total student enrollment?", "", QueryDifficulty.SIMPLE, "SELECT SUM(frpm.enrollment) FROM frpm"),
    ("mb_02", "schools_mini", "What is the maximum sat_score in frpm?", "", QueryDifficulty.SIMPLE, "SELECT MAX(frpm.sat_score) FROM frpm"),
    ("mb_03", "schools_mini", "How many districts are in state California?", "", QueryDifficulty.SIMPLE, "SELECT COUNT(*) FROM districts WHERE districts.state = 'California'"),
    ("mb_04", "finance_mini", "What is the total principal loan_amount?", "", QueryDifficulty.SIMPLE, "SELECT SUM(loans.loan_amount) FROM loans"),
    ("mb_05", "finance_mini", "What is the average credit_score in loans?", "", QueryDifficulty.SIMPLE, "SELECT AVG(loans.credit_score) FROM loans"),
    ("mb_06", "finance_mini", "How many clients are in state New York?", "", QueryDifficulty.SIMPLE, "SELECT COUNT(*) FROM clients WHERE clients.state = 'New York'"),
    ("mb_07", "clinical_mini", "What is the total enrolled_patients in trials?", "", QueryDifficulty.SIMPLE, "SELECT SUM(trials.enrolled_patients) FROM trials"),
    ("mb_08", "clinical_mini", "What is the maximum adverse_events in trials?", "", QueryDifficulty.SIMPLE, "SELECT MAX(trials.adverse_events) FROM trials"),
    # --- MODERATE TIER (8 tasks: FK bridges, abbreviations & NULL ASC) ---
    # mb_09..mb_16 test opaque 2-hop FK bridges ('school_bridge', 'account_disp',
    # 'site_link'), 'Calif' grounding, and SQLite NULL-first ASC LIMIT 1 traps.
    (
        "mb_09",
        "schools_mini",
        "What is the total free_meals for districts in state California?",
        "",
        QueryDifficulty.MODERATE,
        "SELECT SUM(frpm.free_meals) FROM districts JOIN school_bridge ON school_bridge.dist_id = districts.dist_id JOIN frpm ON frpm.cds_id = school_bridge.cds_id WHERE districts.state = 'California'",
    ),
    (
        "mb_10",
        "schools_mini",
        "What is the lowest sat_score in frpm for top 1?",
        "",
        QueryDifficulty.MODERATE,
        "SELECT frpm.sat_score FROM frpm WHERE frpm.sat_score IS NOT NULL ORDER BY frpm.sat_score ASC LIMIT 1",
    ),
    (
        "mb_11",
        "schools_mini",
        "List the top 2 sat_score in frpm.",
        "",
        QueryDifficulty.MODERATE,
        "SELECT frpm.sat_score FROM frpm ORDER BY frpm.sat_score DESC LIMIT 2",
    ),
    (
        "mb_12",
        "finance_mini",
        "What is the total loan_amount for clients in Calif?",
        "",
        QueryDifficulty.MODERATE,
        "SELECT SUM(loans.loan_amount) FROM clients JOIN account_disp ON account_disp.client_id = clients.client_id JOIN loans ON loans.acct_id = account_disp.acct_id WHERE clients.state = 'California'",
    ),
    (
        "mb_13",
        "finance_mini",
        "What is the lowest credit_score in loans for top 1?",
        "",
        QueryDifficulty.MODERATE,
        "SELECT loans.credit_score FROM loans WHERE loans.credit_score IS NOT NULL ORDER BY loans.credit_score ASC LIMIT 1",
    ),
    (
        "mb_14",
        "finance_mini",
        "List the top 3 loan_amount in loans.",
        "",
        QueryDifficulty.MODERATE,
        "SELECT loans.loan_amount FROM loans ORDER BY loans.loan_amount DESC LIMIT 3",
    ),
    (
        "mb_15",
        "clinical_mini",
        "What is the total cured_patients for sponsors in Calif?",
        "",
        QueryDifficulty.MODERATE,
        "SELECT SUM(trials.cured_patients) FROM sponsors JOIN site_link ON site_link.sponsor_id = sponsors.sponsor_id JOIN trials ON trials.site_id = site_link.site_id WHERE sponsors.state = 'California'",
    ),
    (
        "mb_16",
        "clinical_mini",
        "What is the lowest adverse_events in trials for top 1?",
        "",
        QueryDifficulty.MODERATE,
        "SELECT trials.adverse_events FROM trials WHERE trials.adverse_events IS NOT NULL ORDER BY trials.adverse_events ASC LIMIT 1",
    ),
    # --- CHALLENGING TIER (8 tasks: REAL division + FK bridge + top-k) ---
    # mb_17..mb_24 combine external evidence ratios (`INTEGER / INTEGER -> 0` if
    # uncast!) with multi-hop FK joins, 'Calif'/'Los Angeles' literals, & top-k.
    # mb_17..mb_19 evaluate `schools_mini` free-meal ratio and per-school top-2.
    (
        "mb_17",
        "schools_mini",
        "What is the overall free meal rate in Calif?",
        "free meal rate = free_meals / enrollment",
        QueryDifficulty.CHALLENGING,
        "SELECT CAST(SUM(frpm.free_meals) AS REAL) / NULLIF(SUM(frpm.enrollment), 0) FROM districts JOIN school_bridge ON school_bridge.dist_id = districts.dist_id JOIN frpm ON frpm.cds_id = school_bridge.cds_id WHERE districts.state = 'California'",
    ),
    (
        "mb_18",
        "schools_mini",
        "What is the overall free meal rate in Los Angeles?",
        "free meal rate = free_meals / enrollment",
        QueryDifficulty.CHALLENGING,
        "SELECT CAST(SUM(frpm.free_meals) AS REAL) / NULLIF(SUM(frpm.enrollment), 0) FROM districts JOIN school_bridge ON school_bridge.dist_id = districts.dist_id JOIN frpm ON frpm.cds_id = school_bridge.cds_id WHERE districts.county = 'Los Angeles'",
    ),
    (
        "mb_19",
        "schools_mini",
        "What is the highest free meal rate for top 2 schools?",
        "free meal rate = free_meals / enrollment",
        QueryDifficulty.CHALLENGING,
        "SELECT CAST(frpm.free_meals AS REAL) / NULLIF(frpm.enrollment, 0) FROM frpm ORDER BY CAST(frpm.free_meals AS REAL) / NULLIF(frpm.enrollment, 0) DESC LIMIT 2",
    ),
    # mb_20..mb_21 evaluate `finance_mini` loan default ratio + top-2 ranking.
    (
        "mb_20",
        "finance_mini",
        "What is the overall default rate in Calif?",
        "default rate = default_count / installment_count",
        QueryDifficulty.CHALLENGING,
        "SELECT CAST(SUM(loans.default_count) AS REAL) / NULLIF(SUM(loans.installment_count), 0) FROM clients JOIN account_disp ON account_disp.client_id = clients.client_id JOIN loans ON loans.acct_id = account_disp.acct_id WHERE clients.state = 'California'",
    ),
    (
        "mb_21",
        "finance_mini",
        "What is the highest default rate for top 2 loans?",
        "default rate = default_count / installment_count",
        QueryDifficulty.CHALLENGING,
        "SELECT CAST(loans.default_count AS REAL) / NULLIF(loans.installment_count, 0) FROM loans ORDER BY CAST(loans.default_count AS REAL) / NULLIF(loans.installment_count, 0) DESC LIMIT 2",
    ),
    # mb_22..mb_24 evaluate `clinical_mini` cure rate + numeric `> 60` filter.
    (
        "mb_22",
        "clinical_mini",
        "What is the overall cure rate in Calif?",
        "cure rate = cured_patients / enrolled_patients",
        QueryDifficulty.CHALLENGING,
        "SELECT CAST(SUM(trials.cured_patients) AS REAL) / NULLIF(SUM(trials.enrolled_patients), 0) FROM sponsors JOIN site_link ON site_link.sponsor_id = sponsors.sponsor_id JOIN trials ON trials.site_id = site_link.site_id WHERE sponsors.state = 'California'",
    ),
    (
        "mb_23",
        "clinical_mini",
        "What is the highest cure rate for top 3 trials?",
        "cure rate = cured_patients / enrolled_patients",
        QueryDifficulty.CHALLENGING,
        "SELECT CAST(trials.cured_patients AS REAL) / NULLIF(trials.enrolled_patients, 0) FROM trials ORDER BY CAST(trials.cured_patients AS REAL) / NULLIF(trials.enrolled_patients, 0) DESC LIMIT 3",
    ),
    # mb_24 tests spoken/written numeric inequality (`enrolled_patients > 60`).
    (
        "mb_24",
        "clinical_mini",
        "What is the overall cure rate where enrolled_patients greater than 60?",
        "cure rate = cured_patients / enrolled_patients",
        QueryDifficulty.CHALLENGING,
        "SELECT CAST(SUM(trials.cured_patients) AS REAL) / NULLIF(SUM(trials.enrolled_patients), 0) FROM trials WHERE trials.enrolled_patients > 60",
    ),
)


def build_mini_bird_clean_tasks() -> tuple[NLQueryTask, ...]:
  """Returns 24 stratified clean NL2SQL tasks spanning 5 BIRD failure modes.

  Returns:
    Tuple of 24 NLQueryTask definitions (8 simple, 8 moderate, 8 challenging).
  """
  # Materialize immutable NLQueryTask contracts from declarative spec tuples.
  return tuple(
      NLQueryTask(qid, db_id, q_text, ev, diff, gold)
      for qid, db_id, q_text, ev, diff, gold in _CLEAN_TASK_SPECS
  )


def build_voice_bird_tasks() -> tuple[NLQueryTask, ...]:
  """Generates 24 spoken Voice-BIRD tasks from `build_mini_bird_clean_tasks`."""
  # Apply deterministic ASR disfluencies, spoken numbers, and state shorthands.
  return tuple(perturb_task_for_voice(t) for t in build_mini_bird_clean_tasks())
