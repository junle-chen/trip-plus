"""Profile-derived soft-preference evaluator."""

from .evaluator import (
    calculate_user_alignment,
    compile_soft_checks,
    evaluate_soft_checks,
)

__all__ = ["calculate_user_alignment", "compile_soft_checks", "evaluate_soft_checks"]
