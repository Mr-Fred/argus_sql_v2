"""Public API facade for Argus-SQL domain contracts and DDL rendering."""

# Re-export contract exceptions.
from argus_sql.contracts.exceptions import ArgusSQLError
from argus_sql.contracts.exceptions import PromptBuildError
from argus_sql.contracts.exceptions import SchemaGraphError
# Re-export compact SQLite DDL prompt renderer.
from argus_sql.contracts.schema_ddl_renderer import render_schema_ddl
# Re-export immutable relational and pipeline dataclasses.
from argus_sql.contracts.types import ColumnSchema
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ExecutionOutcome
from argus_sql.contracts.types import ForeignKeyEdge
from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import InvariantIssue
from argus_sql.contracts.types import LinkedSchemaContext
from argus_sql.contracts.types import NLQueryTask
from argus_sql.contracts.types import PipelinePrediction
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.contracts.types import SQLCandidate
from argus_sql.contracts.types import TableSchema
from argus_sql.contracts.types import ValueMatch
from argus_sql.contracts.types import VerifiedCandidate

# Explicit symbol export list for static analysis tools.
__all__ = [
    "ArgusSQLError",
    "ColumnSchema",
    "DatabaseSchema",
    "ExecutionOutcome",
    "ForeignKeyEdge",
    "IntentFrame",
    "InvariantIssue",
    "LinkedSchemaContext",
    "NLQueryTask",
    "PipelinePrediction",
    "PromptBuildError",
    "QueryDifficulty",
    "SQLCandidate",
    "SchemaGraphError",
    "TableSchema",
    "ValueMatch",
    "VerifiedCandidate",
    "render_schema_ddl",
]
