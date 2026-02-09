from __future__ import annotations

import math


OPERATION_EXPANSIONS = {
    "==": lambda a, b, eps: [
        "/",
        ["abs", ["-", a, b]],
        ["+", ["abs", ["-", a, b]], eps],
    ],
    "!=": lambda a, b, eps: ["-", 1, ["==", a, b]],
    "=0": lambda a, eps: ["/", ["abs", a], ["+", ["abs", a], eps]],
    ">": lambda a, b, eps: [
        "/",
        ["+", ["-", a, b], ["abs", ["-", a, b]]],
        ["+", ["*", 2, ["abs", ["-", a, b]]], eps],
    ],
    ">=": lambda a, b, eps: [
        "/",
        ["-", ["+", ["-", a, b], ["abs", ["-", a, b]]], eps],
        ["+", ["*", 2, ["abs", ["-", a, b]]], eps],
    ],
    "<": lambda a, b, eps: [
        "/",
        ["+", ["-", b, a], ["abs", ["-", b, a]]],
        ["+", ["*", 2, ["abs", ["-", b, a]]], eps],
    ],
    "<=": lambda a, b, eps: [
        "/",
        ["-", ["+", ["-", b, a], ["abs", ["-", b, a]]], eps],
        ["+", ["*", 2, ["abs", ["-", b, a]]], eps],
    ],
    "!": lambda a: ["-", 1, a],
    "sign": lambda a, eps: ["/", a, ["+", ["abs", a], eps]],
    "max": lambda a, b: ["/", ["+", ["+", a, b], ["abs", ["-", a, b]]], 2],
    "max0": lambda a: ["/", ["+", a, ["abs", a]], 2],
    "min": lambda a, b: ["/", ["-", ["+", a, b], ["abs", ["-", a, b]]], 2],
    "min0": lambda a: ["/", ["-", a, ["abs", a]], 2],
    "if": lambda cond, then_val, else_val: [
        "+",
        ["*", then_val, cond],
        ["*", else_val, ["-", 1, cond]],
    ],
    "mod": lambda a, b: [
        "/",
        ["*", b, ["arctan", ["tan", ["/", ["*", "pi", a], b]]]],
        "pi",
    ],
    "frac": lambda a: ["/", ["arctan", ["tan", ["*", "pi", a]]], "pi"],
    "floor": lambda a: [
        "-",
        ["-", a, 0.5],
        ["/", ["arctan", ["tan", ["+", ["*", "pi", a], ["*", 0.5, "pi"]]]], "pi"],
    ],
    "ceil": lambda a: [
        "+",
        ["+", a, 0.5],
        ["/", ["arctan", ["tan", ["+", ["*", "pi", a], ["*", 0.5, "pi"]]]], "pi"],
    ],
    "round": lambda a: ["-", a, ["/", ["arctan", ["tan", ["*", "pi", a]]], "pi"]],
}

INFIX_OPERATIONS = {"==", "!=", ">", "<", ">=", "<=", "+", "-", "*", "/", "**"}

SYMPY_WORKERS = 4

precedence = {
    "==": 0,
    "!=": 0,
    ">": 0,
    "<": 0,
    ">=": 0,
    "<=": 0,
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
    "**": 3,
}

right_assoc = {"**", "+", "-", "*", "/"}

math_funcs = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "arcsin": math.asin,
    "arccos": math.acos,
    "arctan": math.atan,
    "sqrt": math.sqrt,
    "abs": abs,
    "pi": math.pi,
    "euler": math.e,
}
