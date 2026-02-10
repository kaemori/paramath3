from __future__ import annotations

from .ast_utils import ast_sort_key, substitute_vars
from .constants import OPERATION_EXPANSIONS


def expand_expression(ast):
    if isinstance(ast, list):
        if not ast:
            return ast
        if not isinstance(ast[0], list) and ast[0] in OPERATION_EXPANSIONS:
            expansion_func = OPERATION_EXPANSIONS[ast[0]]
            args = expansion_func.__code__.co_varnames

            expanded_args = [expand_expression(arg) for arg in ast[1:]]
            if "eps" in args:
                return expansion_func(*expanded_args, "epsilon")
            else:
                return expansion_func(*expanded_args)
        else:
            return [expand_expression(elem) for elem in ast]
    else:
        return ast


def expand_functions(ast, functions):
    if isinstance(ast, list):
        if not isinstance(ast[0], list):
            for func in functions:
                if ast[0] == func.name:
                    if len(ast[1:]) != len(func.params):
                        raise ValueError(
                            f"Function {func.name} expects {len(func.params)} arguments, got {len(ast[1:])}"
                        )
                    expanded_args = [
                        expand_functions(arg, functions) for arg in ast[1:]
                    ]
                    substituted = substitute_vars(
                        func.body, dict(zip(func.params, expanded_args))
                    )
                    return expand_functions(substituted, functions)

        return [expand_functions(elem, functions) for elem in ast]
    else:
        return ast


def normalize_ast(ast):
    if not isinstance(ast, list):
        return ast

    if not ast:
        return ast

    ast = [normalize_ast(elem) for elem in ast]

    op = ast[0] if isinstance(ast[0], str) else None

    if op in {"+", "*"}:
        operands = ast[1:]
        operands.sort(key=ast_sort_key, reverse=True)
        return [op] + operands

    return ast
