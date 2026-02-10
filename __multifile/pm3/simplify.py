from __future__ import annotations

import builtins

from .ast_utils import is_constant
from .codegen import generate_expression
from .constants import math_funcs


def simplify_literals_ast(ast, config):
    if isinstance(ast, list):
        simplified = [simplify_literals_ast(elem, config) for elem in ast]
        if is_constant(simplified):
            expr_str = generate_expression(simplified)
            eval_namespace = {
                "__builtins__": builtins.__dict__,
                **math_funcs,
                "epsilon": config.epsilon,
            }
            result = eval(expr_str, eval_namespace)
            if isinstance(result, float):
                result = round(result, config.precision)
            return result
        else:
            return simplified
    else:
        return ast


def simplify_algebratic_identities(ast):
    if isinstance(ast, list):
        if ast[0] == "*":
            if len(ast) == 3:
                if ast[1] == 0 or ast[2] == 0:
                    return 0
                if ast[1] == 1:
                    return simplify_algebratic_identities(ast[2])
                if ast[2] == 1:
                    return simplify_algebratic_identities(ast[1])
                if ast[1] == -1:
                    return ["-", simplify_algebratic_identities(ast[2])]
                if ast[2] == -1:
                    return ["-", simplify_algebratic_identities(ast[1])]
                if ast[1] == ast[2]:
                    return ["**", simplify_algebratic_identities(ast[1]), 2]
        if ast[0] == "+":
            if len(ast) == 3:
                if ast[1] == 0:
                    return simplify_algebratic_identities(ast[2])
                if ast[2] == 0:
                    return simplify_algebratic_identities(ast[1])
                if ast[1] == ["-", simplify_algebratic_identities(ast[2])]:
                    return 0
                if ast[2] == ["-", simplify_algebratic_identities(ast[1])]:
                    return 0
                if ast[1] == ast[2]:
                    return ["*", 2, simplify_algebratic_identities(ast[1])]
        if ast[0] == "-":
            if len(ast) == 3:
                if ast[2] == 0:
                    return simplify_algebratic_identities(ast[1])
                if ast[1] == ast[2]:
                    return 0
        if ast[0] == "/":
            if len(ast) == 3:
                if ast[1] == 0:
                    return 0
                if ast[2] == 1:
                    return simplify_algebratic_identities(ast[1])
                if ast[1] == ast[2]:
                    return 1
        if ast[0] == "**":
            if len(ast) == 3:
                if ast[1] == 0 and ast[2] != 0:
                    return 0
                if ast[2] == 0:
                    return 1
                if ast[2] == 1:
                    return simplify_algebratic_identities(ast[1])

        return ast

    else:
        return ast


def simplify_ast(ast, config):
    ast = simplify_literals_ast(ast, config)
    ast = simplify_algebratic_identities(ast)
    return ast
