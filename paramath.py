import math
import builtins
from dataclasses import dataclass
from typing import Union


OPERATION_EXPANSIONS = {
    "==": lambda a, b, eps: [
        "/",
        ["abs", [a, "-", b]],
        [[["abs", [a, "-", b]], "+", eps]],
    ],
    "=0": lambda a, eps: [
        "/",
        ["abs", a],
        [["abs", a], "+", eps],
    ],
    ">": lambda a, b, eps: [
        "/",
        [[a, "-", b], "+", ["abs", [a, "-", b]]],
        [[[2, "*", ["abs", [a, "-", b]]], "+", eps]],
    ],
    ">=": lambda a, b, eps: [
        "/",
        [[[a, "-", b], "+", ["abs", [a, "-", b]]], "-", eps],
        [[2, "*", ["abs", [a, "-", b]]], "+", eps],
    ],
    "<": lambda a, b, eps: [
        "/",
        [[b, "-", a], "+", ["abs", [b, "-", a]]],
        [[2, "*", ["abs", [b, "-", a]]], "+", eps],
    ],
    "<=": lambda a, b, eps: [
        "/",
        [[[b, "-", a], "+", ["abs", [b, "-", a]]], "-", eps],
        [[2, "*", ["abs", [b, "-", a]]], "+", eps],
    ],
    "!": lambda a: [1, "-", a],
    "sign": lambda a, eps: [
        "/",
        a,
        [["abs", a], "+", eps],
    ],
    "max": lambda a, b: [
        "/",
        [[a, "+", b], "+", ["abs", [a, "-", b]]],
        2,
    ],
    "max0": lambda a: [
        "/",
        [a, "+", ["abs", a]],
        2,
    ],
    "min": lambda a, b: [
        "/",
        [[a, "+", b], "-", ["abs", [a, "-", b]]],
        2,
    ],
    "min0": lambda a: [
        "/",
        [a, "-", ["abs", a]],
        2,
    ],
    "if": lambda cond, then_val, else_val: [
        [then_val, "*", cond],
        "+",
        [else_val, "*", [1, "-", cond]],
    ],
    "mod": lambda a, b: [
        "/",
        [b, "*", ["arctan", ["tan", [["pi", "*", a], "/", b]]]],
        "pi",
    ],
    "frac": lambda a: [
        "/",
        ["arctan", ["tan", ["pi", "*", a]]],
        "pi",
    ],
    "floor": lambda a: [
        [a, "-", 0.5],
        "-",
        ["/", ["arctan", ["tan", [["pi", "*", a], "+", [0.5, "*", "pi"]]]], "pi"],
    ],
    "ceil": lambda a: [
        [a, "+", 0.5],
        "+",
        ["/", ["arctan", ["tan", [["pi", "*", a], "+", [0.5, "*", "pi"]]]], "pi"],
    ],
    "round": lambda a: [
        a,
        "-",
        ["/", ["arctan", ["tan", ["pi", "*", a]]], "pi"],
    ],
}
INFIX_OPERATIONS = {
    "==",
    "=0",
    "!",
    ">",
    "<",
    ">=",
    "<=",
    "+",
    "-",
    "*",
    "/",
    "**",
    "mod",
}

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
    "abs": abs,
    "pi": math.pi,
    "e": math.e,
}


def num(val: Union[str, int, float]) -> Union[int, float]:
    try:
        if float(val).is_integer():
            return int(val)
        else:
            return float(val)
    except ValueError:
        raise ValueError(f"Cannot turn {val!r} into a number")


class EvalVars:
    def __init__(self, d):
        self.__dict__.update(d)


@dataclass
class Output:
    output: str = None
    var: str | None = None


@dataclass
class ProgramConfig:
    precision: int = None
    epsilon: float = None
    variables: list = None
    dupe: bool = None
    simplify: bool = None
    sympy: bool = None
    output: str = None


@dataclass
class FunctionDef:
    name: str
    params: list
    body: list
    config: ProgramConfig


def tokenize(line):
    special_tokens = {"    ": "TAB", "\t": "TAB"}
    multi_char_ops = [key for key in OPERATION_EXPANSIONS.keys() if len(key) > 1]
    multi_char_ops += [":=", "//", "**"]

    for token, replacement in special_tokens.items():
        line = line.replace(token, f" __{replacement}__ ")

    for i, op in enumerate(multi_char_ops):
        line = line.replace(op, f" __MULTIOP_{i}__ ")

    splitters = ["(", ")", ",", "+", "-", "*", "/", "=", "\n"]
    for splitter in splitters:
        line = line.replace(splitter, f" {splitter} ")
    for i, op in enumerate(multi_char_ops):
        line = line.replace(f"__MULTIOP_{i}__", op)

    while "  " in line:
        line = line.replace("  ", " ")

    return line.lstrip().split(" ")


def generate_ast(tokens):
    tokens = [t for t in tokens if t and t != "\n"]

    def parse_expr(idx):
        result = []
        i = idx

        while i < len(tokens):
            token = tokens[i]

            if token == ")":
                return result, i
            elif token == ",":
                return result, i
            elif i + 1 < len(tokens) and tokens[i + 1] == "(":
                func_name = token
                args = []
                i += 2

                while i < len(tokens) and tokens[i] != ")":
                    if tokens[i] == ",":
                        i += 1
                        continue
                    arg, i = parse_expr(i)
                    args.append(arg if len(arg) != 1 else arg[0])

                result.append([func_name] + args)
                i += 1
            elif token == "(":
                i += 1
                inner, i = parse_expr(i)
                result.append(inner if len(inner) != 1 else inner[0])
                i += 1
            else:
                result.append(token)
                i += 1

        return result, i

    ast, _ = parse_expr(0)
    return ast if len(ast) != 1 else ast[0]


def infix_to_postfix(ast):
    if isinstance(ast, list):
        if len(ast) > 1 and isinstance(ast[1], str) and ast[1] in INFIX_OPERATIONS:
            op = ast[1]
            left = infix_to_postfix(ast[0])
            right = infix_to_postfix(ast[2])
            return [op, left, right]
        else:
            return [infix_to_postfix(elem) for elem in ast]
    else:
        return ast


def expand_expression(ast, eps):
    if isinstance(ast, list):
        if not ast:
            return ast
        if not isinstance(ast[0], list) and ast[0] in OPERATION_EXPANSIONS:
            expansion_func = OPERATION_EXPANSIONS[ast[0]]
            args = expansion_func.__code__.co_varnames

            expanded_args = [expand_expression(arg, eps) for arg in ast[1:]]
            if "eps" in args:
                return expansion_func(*expanded_args, eps)
            else:
                return expansion_func(*expanded_args)
        else:
            return [expand_expression(elem, eps) for elem in ast]
    else:
        return ast


def parse_pragma(line: str, config: ProgramConfig):
    pragma, value = line.split(" ", 1)
    if pragma == "precision":
        config.precision = int(value)
    elif pragma == "epsilon":
        config.epsilon = float(value)
    elif pragma == "variables":
        config.variables = [var.strip() for var in value.split(" ")]
    elif pragma == "dupe":
        config.dupe = value.lower() == "true"
    elif pragma == "simplify":
        config.simplify = value.lower() == "true"
    elif pragma == "sympy":
        config.sympy = value.lower() == "true"
    else:
        raise ValueError(f"Unknown pragma: {pragma}")


def substitute_vars(tokens, subst_vars):
    for i in range(len(tokens)):
        if isinstance(tokens[i], list):
            substitute_vars(tokens[i], subst_vars)
        elif tokens[i] in subst_vars:
            tokens[i] = subst_vars[tokens[i]]
    return tokens


def parse_pm3(code):
    config = ProgramConfig()
    subst_vars = {}

    collecting_function = False
    collecting_loop = False

    funcs = []

    for line_num, line in enumerate(code):
        line = line.split("#", 1)[0].rstrip()
        if not line:
            continue

        tokens = tokenize(line)
        # print(line, tokens)

        if collecting_function:
            if not tokens[0] == "__TAB__":
                collecting_function = False
                funcs.append(current_function)
            else:
                current_function.body.append(tokens[1:])
                continue

        if collecting_loop:
            if not tokens[0] == "__TAB__":
                collecting_loop = False
            else:
                continue  # tbi

        if not collecting_function or not collecting_loop:
            if tokens[0] == "__TAB__":
                raise ValueError(
                    f"Unexpected indentation outside function or loop at line {line_num + 1}"
                )

            if tokens[0] == "def":
                if collecting_loop:
                    raise ValueError(
                        f"Function definitions inside loops are not supported at line {line_num + 1}"
                    )
                collecting_function = True
                name = tokens[1][:-1]
                params = [
                    param.strip() for param in line[3:].strip().split(":")[1].split(",")
                ]
                current_function = FunctionDef(
                    name=name, params=params, body=[], config=config
                )

            elif tokens[0] == "repeat":
                collecting_loop = True
                # tbd

            elif tokens[0] == "out":
                output = tokens[1]
                if output in config.variables:
                    config.output = output
                else:
                    raise ValueError(
                        f"Output variable {output!r} not declared in variables at line {line_num + 1}"
                    )

            elif tokens[0] == "return":
                expr_tokens = tokens[1:]

                result = substitute_vars(expr_tokens, subst_vars)
                ast = generate_ast(result)
                ast = infix_to_postfix(ast)
                expr = expand_expression(ast, eps=config.epsilon)
                print(expr)
                config.output = None

            elif tokens[0] == "//":
                if not collecting_function and not collecting_loop:
                    parse_pragma(line.strip()[2:].strip(), config)
                else:
                    raise ValueError(
                        f"Pragmas inside functions or loops are not supported at line {line_num + 1}"
                    )

            else:
                # non-function, non-pragma line
                if tokens[1] == "=":
                    subst_vars[tokens[0]] = tokens[2:]
                if tokens[1] == ":=":
                    # python eval!
                    var_name = tokens[0]
                    expr = line.split(":=")[1].strip()
                    eval_namespace = {
                        "__builtins__": builtins.__dict__,
                        "vars": EvalVars(subst_vars),
                        **math_funcs,
                    }
                    result = eval(expr, eval_namespace)
                    subst_vars[var_name] = [str(result)]

    # print(config)
    # print(subst_vars)
    # print(funcs)


if __name__ == "__main__":
    with open("mockup.pm3") as f:
        code = [line.rstrip() for line in f.readlines() if line]
        parse_pm3(code)
