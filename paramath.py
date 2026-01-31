import math
import builtins
from dataclasses import dataclass
from typing import Union

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


@dataclass
class RepeatLoop:
    itr_var: str
    times: Union[int, str]
    body: list


def is_constant(expr):
    if isinstance(expr, list):
        return all(is_constant(subexpr) for subexpr in expr[1:])
    else:
        try:
            num(expr)
            return True
        except ValueError:
            return False


def tokenize(line):
    special_tokens = {"    ": "TAB", "\t": "TAB"}
    multi_char_ops = [key for key in INFIX_OPERATIONS if len(key) > 1]
    multi_char_ops += [":=", "//", "**"]

    for token, replacement in special_tokens.items():
        line = line.replace(token, f" __{replacement}__ ")

    for i, op in enumerate(multi_char_ops):
        line = line.replace(op, f" __MULTIOP_{i}__ ")

    splitters = ["(", ")", ",", "+", "-", "*", "/", "=", "!", "\n"]
    for splitter in splitters:
        line = line.replace(splitter, f" {splitter} ")
    for i, op in enumerate(multi_char_ops):
        line = line.replace(f"__MULTIOP_{i}__", op)

    while "  " in line:
        line = line.replace("  ", " ")

    return line.strip().split(" ")


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
    right_assoc = {"**"}

    def is_op(token):
        return isinstance(token, str) and token in INFIX_OPERATIONS

    def convert(node):
        if not isinstance(node, list):
            return node

        if not node:
            return node

        if len(node) > 1 and isinstance(node[1], str) and node[1] in INFIX_OPERATIONS:
            items = node

            i = 0

            def peek():
                return items[i] if i < len(items) else None

            def consume():
                nonlocal i
                tok = items[i]
                i += 1
                return tok

            def parse_atom():
                tok = peek()
                if tok in ("+", "-"):
                    op = consume()
                    rhs = parse_atom()
                    if op == "+":
                        return rhs
                    return ["-", 0, rhs]
                if tok == "!":
                    consume()
                    rhs = parse_atom()
                    return ["!", rhs]

                tok = consume()
                if isinstance(tok, list):
                    return convert(tok)
                return tok

            def parse_expr(min_prec=0):
                left = parse_atom()
                while True:
                    op = peek()
                    if not is_op(op) or op not in precedence:
                        break
                    prec = precedence[op]
                    if prec < min_prec:
                        break

                    consume()
                    next_min = prec if op in right_assoc else prec + 1
                    right = parse_expr(next_min)
                    left = [op, left, right]

                return left

            result = parse_expr(0)
            if i != len(items):
                leftover = " ".join(str(x) for x in items[i:])
                raise SyntaxError(
                    f"Could not fully parse expression, leftover tokens: {leftover}"
                )
            return result

        if isinstance(node[0], str):
            return [node[0]] + [convert(arg) for arg in node[1:]]

        return [convert(elem) for elem in node]

    return convert(ast)


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
    if isinstance(tokens, list):
        return [substitute_vars(token, subst_vars) for token in tokens]
    elif tokens in subst_vars:
        return subst_vars[tokens]
    else:
        return tokens


def parse_expression(tokens, subst_vars, functions):
    result = substitute_vars(tokens, subst_vars)
    ast = generate_ast(result)
    ast = infix_to_postfix(ast)
    expr = expand_expression(ast)
    expr = expand_functions(expr, functions)
    return expr


def parse_function(func_tokens, lines, subst_vars, functions):
    local_scope = subst_vars.copy()
    for line_no, tokens in enumerate(func_tokens):
        if tokens[1] == "=":
            local_scope[tokens[0]] = parse_expression(
                tokens[2:], local_scope, functions
            )
        elif tokens[1] == ":=":
            var_name = tokens[0]
            expr = lines[line_no].split(":=", 1)[1].strip()
            eval_namespace = {
                "__builtins__": builtins.__dict__,
                "vars": EvalVars(local_scope),
                **math_funcs,
            }
            result = eval(expr, eval_namespace)
            local_scope[var_name] = [str(result)]

        elif tokens[0] == "return":
            expr_tokens = tokens[1:]
            return parse_expression(expr_tokens, local_scope, functions)

        else:
            raise SyntaxError(f"Unknown statement in function: {' '.join(tokens)}")


def parse_repeat(
    body_tokens,
    lines,
    times,
    repeat_var,
    subst_vars,
    variables,
    functions,
    *,
    outer_bindings=None,
):
    if outer_bindings is None:
        outer_bindings = {}

    variables = variables or []

    if repeat_var is not None and (
        repeat_var in variables
        or repeat_var in subst_vars.keys()
        or repeat_var in outer_bindings
    ):
        raise ValueError(
            f"Iteration variable {repeat_var!r} conflicts with declared variables"
        )

    def dedent_once(token_list):
        if token_list and token_list[0] == "__TAB__":
            return token_list[1:]
        return token_list

    def strip_all_tabs(token_list):
        while token_list and token_list[0] == "__TAB__":
            token_list = token_list[1:]
        return token_list

    results = []
    for itr in range(num(times)):
        itr_bindings = (
            outer_bindings
            if repeat_var is None
            else outer_bindings | {repeat_var: str(itr)}
        )

        line_no = 0
        while line_no < len(body_tokens):
            tokens = strip_all_tabs(body_tokens[line_no])
            if not tokens:
                line_no += 1
                continue

            if len(tokens) > 1 and tokens[1] == "=":
                subst_vars[tokens[0]] = parse_expression(
                    tokens[2:], subst_vars | itr_bindings, functions
                )
                line_no += 1
                continue

            if len(tokens) > 1 and tokens[1] == ":=":
                var_name = tokens[0]
                expr = lines[line_no].split(":=", 1)[1].strip()
                eval_namespace = {
                    "__builtins__": builtins.__dict__,
                    "vars": EvalVars(subst_vars | itr_bindings),
                    **math_funcs,
                }
                result = eval(expr, eval_namespace)
                subst_vars[var_name] = [str(result)]
                line_no += 1
                continue

            if tokens[0] == "repeat":
                if len(tokens) < 2:
                    raise SyntaxError("repeat requires at least a repetition count")

                nested_times = tokens[1]
                nested_repeat_var = tokens[2] if len(tokens) >= 3 else None

                nested_body_tokens = []
                nested_body_lines = []
                next_line_no = line_no + 1

                while next_line_no < len(body_tokens) and lines[
                    next_line_no
                ].startswith("    "):
                    nested_body_lines.append(lines[next_line_no][4:])
                    nested_body_tokens.append(dedent_once(body_tokens[next_line_no]))
                    next_line_no += 1

                results.extend(
                    parse_repeat(
                        nested_body_tokens,
                        nested_body_lines,
                        nested_times,
                        nested_repeat_var,
                        subst_vars,
                        variables,
                        functions,
                        outer_bindings=itr_bindings,
                    )
                )

                line_no = next_line_no
                continue

            if tokens[0] == "return":
                expr_tokens = tokens[1:]
                results.append(
                    parse_expression(expr_tokens, subst_vars | itr_bindings, functions)
                )
                line_no += 1
                continue

            raise SyntaxError(f"Unknown statement in repeat: {' '.join(tokens)}")

    return results


def parse_pm3(code):
    config = ProgramConfig()
    subst_vars = {}

    collecting_function = False
    collecting_loop = False

    funcs = []
    compiled = []
    for line_num, line in enumerate(code):
        line = line.split("#", 1)[0].rstrip()
        if not line:
            continue

        tokens = tokenize(line)
        # print(line, tokens)

        if collecting_function:
            if not tokens[0] == "__TAB__":
                collecting_function = False
                current_function.body = parse_function(
                    function_tokens, function_body, subst_vars, funcs
                )
                funcs.append(current_function)
            else:
                function_body.append(line[4:])
                function_tokens.append(tokens[1:])
                continue

        if collecting_loop:
            if not tokens[0] == "__TAB__":
                collecting_loop = False
                repeat_result = parse_repeat(
                    repeat_tokens,
                    repeat_body,
                    repeat_times,
                    repeat_var,
                    subst_vars,
                    config.variables,
                    funcs,
                )
                compiled.extend(repeat_result)
            else:
                repeat_body.append(line[4:])
                repeat_tokens.append(tokens[1:])

        if not collecting_function and not collecting_loop:
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
                function_body = []
                function_tokens = []
                function_name = tokens[1]
                params = tokens[2:]
                current_function = FunctionDef(
                    name=function_name, params=params, body=[]
                )
            elif tokens[0] == "repeat":
                collecting_loop = True
                repeat_body = []
                repeat_tokens = []
                repeat_times = tokens[1]
                repeat_var = tokens[2] if len(tokens) >= 3 else None

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
                expr = parse_expression(expr_tokens, subst_vars, funcs)
                compiled.append(expr)
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

    if collecting_function:
        current_function.body = parse_function(
            function_tokens, function_body, subst_vars, funcs
        )
        funcs.append(current_function)

    if collecting_loop:
        repeat_result = parse_repeat(
            repeat_tokens,
            repeat_body,
            repeat_times,
            repeat_var,
            subst_vars,
            config.variables,
            funcs,
        )
        compiled.extend(repeat_result)

    # print(config)
    # print(subst_vars)
    # print(funcs)

    for item in compiled:
        print(item)


if __name__ == "__main__":
    with open("mockup.pm3") as f:
        code = [line.rstrip() for line in f.readlines() if line]
        parse_pm3(code)
