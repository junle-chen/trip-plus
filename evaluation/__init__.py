"""Evaluation tools for structured travel-planning outputs."""

__all__ = ["convert_reports", "evaluate_plans"]


def __getattr__(name):
    if name == "convert_reports":
        from .conversion import convert_reports

        return convert_reports
    if name == "evaluate_plans":
        from .single_turn import evaluate_plans

        return evaluate_plans
    raise AttributeError(name)
