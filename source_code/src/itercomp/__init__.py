"""IterCOMP — nén câu lệnh thích ứng có nhận thức suy luận cho hỏi đáp đa bước.

Tái hiện ACL 2026 (https://aclanthology.org/2026.acl-long.1559/) và mở rộng
đánh giá sang tiếng Việt.
"""

from .core import IterCompResult, Segment, decompose, itercomp
from .budget import budget_filter, split_long_segments
from .data import DATA, load_dataset
from .fertility import (count_broken_syllables, effective_ratio,
                        measure_fertility, relative_fertility)
from .llm import make_llm
from .metrics import (exact_match, f1_score, f1_span, normalize,
                      normalize_boolean, tokenize)
from .reader import clean_answer, make_reader
from .scorer import make_scorer, percentile_filter
from .stopping import should_stop, evidence_confidence, marginal_gain
from .stats import (bootstrap_ci, min_detectable_diff, paired_bootstrap,
                    required_n)

__all__ = [
    "itercomp", "IterCompResult", "Segment", "decompose",
    "load_dataset", "DATA",
    "budget_filter", "split_long_segments",
    "measure_fertility", "relative_fertility", "effective_ratio",
    "count_broken_syllables",
    "make_llm", "make_reader", "clean_answer",
    "make_scorer", "percentile_filter",
    "should_stop", "evidence_confidence", "marginal_gain",
    "bootstrap_ci", "paired_bootstrap", "min_detectable_diff", "required_n",
    "exact_match", "f1_score", "f1_span", "normalize", "normalize_boolean",
    "tokenize",
]
