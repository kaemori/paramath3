from __future__ import annotations

from .utils import num, stable_serialize


def is_constant(expr):
    if isinstance(expr, list):
        return all(is_constant(subexpr) for subexpr in expr[1:])
    else:
        if expr in ("epsilon", "pi", "euler"):
            return True
        try:
            num(expr)
            return True
        except ValueError:
            return False


def fix_structure(ast):
    if isinstance(ast, list):
        if len(ast) == 1:
            return fix_structure(ast[0])
        else:
            return [fix_structure(elem) for elem in ast]
    else:
        return ast


def substitute_vars(tokens, subst_vars):
    if isinstance(tokens, list):
        return [substitute_vars(token, subst_vars) for token in tokens]
    elif tokens in subst_vars:
        return subst_vars[tokens]
    else:
        return tokens


def expr_length(ast) -> int:
    from .codegen import generate_expression

    return len(generate_expression(ast).replace(" ", ""))


def ast_sort_key(x):
    is_const = is_constant(x)
    try:
        length = expr_length(x) if isinstance(x, list) else (0 if is_const else 9999)
    except Exception:
        length = 9999
    s = stable_serialize(x)
    return (0 if is_const else 1, length, s)
