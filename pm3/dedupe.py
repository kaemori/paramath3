from __future__ import annotations

from .ast_utils import expr_length
from .utils import stable_serialize


def all_sublists(expr):
    if isinstance(expr, list):
        yield expr
        for item in expr:
            yield from all_sublists(item)


def replace_in_ast(ast, pattern, replacement: str):
    if ast == pattern:
        return replacement
    if isinstance(ast, list):
        return [replace_in_ast(x, pattern, replacement) for x in ast]
    return ast


def find_beneficial_duplicates(ast, min_length: int = 4, min_savings: int = -999):
    seen = {}

    if not isinstance(ast, list):
        return []

    for sub in all_sublists(ast):
        key = stable_serialize(sub)
        seen.setdefault(key, {"count": 0, "obj": sub})
        seen[key]["count"] += 1

    beneficial_dupes = []
    for data in seen.values():
        if data["count"] > 1:
            dupe_length = expr_length(data["obj"])
            if dupe_length >= min_length:
                total_size = data["count"] * dupe_length
                savings = total_size - (dupe_length + 3 + (data["count"] - 1) * 3)
                if savings > min_savings:
                    beneficial_dupes.append((savings, total_size, data["obj"]))

    beneficial_dupes.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return beneficial_dupes


def extract_subexpressions(ast, output_variable, min_savings):
    current_ast = ast
    beneficial_dupes = find_beneficial_duplicates(current_ast, min_savings=min_savings)

    if not beneficial_dupes:
        return [(current_ast, output_variable)]
    largest_dupe = beneficial_dupes[0][2]
    replaced_ast = replace_in_ast(current_ast, largest_dupe, "ans")
    return [(largest_dupe, "ans"), (replaced_ast, output_variable)]
