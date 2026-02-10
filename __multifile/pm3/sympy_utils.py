from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

from .constants import SYMPY_WORKERS
from .logging import print_debug, print_verbose
from .tokenizer import tokenize


def _normalize_for_sympy(expr: str) -> str:
    return (
        expr.replace("arcsin", "asin")
        .replace("arccos", "acos")
        .replace("arctan", "atan")
        .replace("abs(", "Abs(")
    )


def _denormalize_from_sympy(expr: str) -> str:
    expr = (
        expr.replace("asin", "arcsin")
        .replace("acos", "arccos")
        .replace("atan", "arctan")
        .replace("Abs", "abs")
    )
    return expr


def _format_expression(expr: str) -> str:
    return " ".join(tokenize(expr)).replace("( ", "(").replace(" )", ")")


def _sympy_simplify_worker(task: tuple[int, str]) -> tuple[int, str]:
    index, expr_str = task
    try:
        import sympy

        sympy_in = _normalize_for_sympy(expr_str)
        sympy_expr = sympy.sympify(sympy_in)
        simplified = sympy.simplify(sympy_expr)
        out = str(simplified).replace(".0e", "e")
        out = _denormalize_from_sympy(out)
        out = _format_expression(out)
        return index, out
    except ImportError:
        raise ImportError(
            "SymPy is required but could not be imported. Try: pip install sympy"
        )
    except Exception:
        return index, expr_str


def simplify_sympy_multicore(
    expressions: list[str], *, show_progress: bool = True
) -> list[str]:
    if not expressions:
        return []

    print_verbose(
        f"sympy: simplifying {len(expressions)} expression(s) (multicore={len(expressions) > 1})"
    )

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    tasks = list(enumerate(expressions))
    results: list[str] = list(expressions)

    if len(tasks) == 1:
        idx, simplified = _sympy_simplify_worker(tasks[0])
        results[idx] = simplified
        return results

    max_workers = max(1, min(len(tasks), SYMPY_WORKERS))
    print_debug(f"sympy: using {max_workers} worker(s)")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_sympy_simplify_worker, task) for task in tasks]
        iterator = as_completed(futures)

        if tqdm is not None and show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="sympy", unit="expr")

        for fut in iterator:
            idx, simplified = fut.result()
            results[idx] = simplified

    return results
