"""Public API facade for Argus-SQL voice canonicalization and schema linking."""

# Re-export linking domain exceptions.
from argus_sql.linking.exceptions import LinkingError
from argus_sql.linking.exceptions import ValueIndexError
# Re-export Steiner FK-closure schema linker.
from argus_sql.linking.fk_schema_linker import link_schema_with_fk_closure
# Re-export spoken disfluency and intent canonicalizer.
from argus_sql.linking.intent_canonicalizer import canonicalize_intent
from argus_sql.linking.llm_intent_extractor import FlashLiteIntentExtractor
from argus_sql.linking.llm_intent_extractor import canonicalize_intent_with_optional_llm
from argus_sql.linking.token_fsm import AhoCorasickAutomaton
from argus_sql.linking.token_fsm import AhoCorasickMatch
from argus_sql.linking.token_fsm import LexToken
from argus_sql.linking.token_fsm import tokenize_utterance
# Re-export linking configuration and score structures.
from argus_sql.linking.types import ColumnScore
from argus_sql.linking.types import IntentExtractorPort
from argus_sql.linking.types import SchemaLinkerConfig
from argus_sql.linking.types import ValueIndexEntry
# Re-export character 3-gram database cell value retriever.
from argus_sql.linking.value_retriever import CellValueRetriever

# Explicit symbol export list.
__all__ = [
    "AhoCorasickAutomaton",
    "AhoCorasickMatch",
    "CellValueRetriever",
    "ColumnScore",
    "FlashLiteIntentExtractor",
    "IntentExtractorPort",
    "LexToken",
    "LinkingError",
    "SchemaLinkerConfig",
    "ValueIndexEntry",
    "ValueIndexError",
    "canonicalize_intent",
    "canonicalize_intent_with_optional_llm",
    "link_schema_with_fk_closure",
    "tokenize_utterance",
]
