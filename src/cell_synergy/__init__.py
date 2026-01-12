"""
Cell Synergy: Multimodal Self-Supervised Learning for Biological Data.
"""

# Import SIS functions for easy access
from .sis import (
    SIS,
    set_RESULTS,
    compute_sis,
    compute_sis_all_models,
    print_sis_summary,
    get_sis_ranking,
    analyze_synergy_patterns
)

__all__ = [
    'SIS',
    'set_RESULTS',
    'compute_sis',
    'compute_sis_all_models',
    'print_sis_summary',
    'get_sis_ranking',
    'analyze_synergy_patterns',
]
