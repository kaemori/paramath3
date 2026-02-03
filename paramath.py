import math
import json
import builtins
import re
from dataclasses import dataclass, replace
from typing import Union


def _capitalize_error_message(message: str) -> str:
    msg = "" if message is None else str(message)
    msg = msg.strip()
    if not msg:
        return msg

    # Uppercase the first letter-like character, without lowercasing the rest.
    for i, ch in enumerate(msg):
        if ch.isalpha():
            return msg[:i] + ch.upper() + msg[i + 1 :]

    # Fallback: just uppercase the first character.
    return msg[0].upper() + msg[1:]


def _format_error(message: str, line_num: int | None = None) -> str:
    msg = _capitalize_error_message(message)

    if line_num is None:
        return msg

    # Avoid doubling up if a line prefix already exists.
    if msg.lower().startswith("line "):
        return msg

    # Strip common trailing "at/in line N" patterns before prefixing.
    for suffix in (
        f" at line {line_num}",
        f" in line {line_num}",
        f" at Line {line_num}",
        f" in Line {line_num}",
    ):
        if msg.endswith(suffix):
            msg = msg[: -len(suffix)].rstrip()

    # Also strip patterns like "... at line N:" (rare but happens).
    msg = re.sub(r"\s+(at|in)\s+line\s+\d+\s*:\s*$", "", msg, flags=re.IGNORECASE)

    return f"Line {line_num}: {msg}"


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
    "euler": math.e,
}


def num(val: Union[str, int, float]) -> Union[int, float]:
    try:
        if float(val).is_integer():
            return int(val)
        else:
            return float(val)
    except ValueError:
        raise ValueError(_format_error(f"Cannot turn {val!r} into a number"))


class EvalVars:
    def __init__(self, d):
        self.__dict__.update(d)


@dataclass
class Output:
    output: str = None
    ast: Union[list, float, int] = None
    config: "ProgramConfig" = None


@dataclass
class ProgramConfig:
    precision: int = None
    epsilon: float = None
    variables: list = None
    dupe: bool = None
    dupe_min_savings: int = 0
    simplify: bool = None
    sympy: bool = None
    output: str = None

    def __post_init__(self):
        if self.variables is None:
            self.variables = list("abcdefxyzm")


@dataclass
class FunctionDef:
    name: str
    params: list
    body: list


@dataclass
class AstOutput:
    output: str = None
    ast: Union[list, float, int] = None


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
            elif (
                i + 1 < len(tokens)
                and tokens[i + 1] == "("
                and isinstance(token, str)
                and token.isidentifier()
            ):
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

    ast, end_idx = parse_expr(0)
    if end_idx != len(tokens):
        bad = tokens[end_idx]
        if bad == ")":
            raise SyntaxError(_format_error("Unexpected closing parenthesis"))
        if bad == ",":
            raise SyntaxError(_format_error("Unexpected comma"))
        raise SyntaxError(_format_error(f"Unexpected token: {bad!r}"))
    ast = ast if len(ast) != 1 else ast[0]
    return ast


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


def ast_sort_key(x):
    is_const = is_constant(x)
    try:
        length = expr_length(x) if isinstance(x, list) else (0 if is_const else 9999)
    except Exception:
        length = 9999
    s = json.dumps(x, sort_keys=False, default=str)
    return (0 if is_const else 1, length, s)


def normalize_ast(ast):
    if not isinstance(ast, list):
        return ast

    if not ast:
        return ast

    ast = [normalize_ast(elem) for elem in ast]

    op = ast[0] if isinstance(ast[0], str) else None

    if op in {"+", "*"}:
        operands = []
        for a in ast[1:]:
            if isinstance(a, list) and a and a[0] == op:
                operands.extend(a[1:])
            else:
                operands.append(a)

        operands.sort(key=ast_sort_key)
        return [op] + operands

    return ast


def parse_pragma(line: str, config: ProgramConfig):
    pragma, value = line.split(" ", 1)
    if pragma == "precision":
        if not value.isdigit():
            raise ValueError(
                _format_error(f"Precision must be an integer, got {value!r}")
            )
        config.precision = int(value)
    elif pragma == "epsilon":
        try:
            float(value)
        except ValueError:
            raise ValueError(
                _format_error(f"Epsilon must be a float, got {value!r}")
            ) from None
        config.epsilon = float(value)
    elif pragma == "variables":
        config.variables = [var.strip() for var in value.split(" ")]
    elif pragma == "dupe":
        parts = value.strip().split()
        if not parts:
            raise ValueError(_format_error("Dupe requires true/false"))
        config.dupe = parts[0].lower() == "true"
        if len(parts) >= 2:
            try:
                config.dupe_min_savings = int(parts[1])
            except ValueError:
                raise ValueError(
                    _format_error("Dupe min_savings must be an integer")
                ) from None
    elif pragma == "simplify":
        config.simplify = value.lower() == "true"
    elif pragma == "sympy":
        config.sympy = value.lower() == "true"
    else:
        raise ValueError(_format_error(f"Unknown pragma: {pragma}"))


def _raise_if_conflicts_with_declared_variables(
    name: str,
    declared_variables: list | None,
    *,
    kind: str,
    line_num: int | None = None,
):
    declared_variables = declared_variables or []
    if name in declared_variables:
        raise ValueError(
            _format_error(
                f"{kind} {name!r} conflicts with a name declared in //variables",
                line_num,
            )
        )


def _is_number_like(token: str) -> bool:
    try:
        num(token)
        return True
    except Exception:
        return False


def _validate_balanced_parentheses(tokens: list, *, line_num: int | None = None):
    depth = 0
    for tok in tokens:
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
            if depth < 0:
                raise SyntaxError(
                    _format_error("Unexpected closing parenthesis", line_num)
                ) from None

    if depth != 0:
        raise SyntaxError(_format_error("Unexpected end of tokens", line_num)) from None


def _validate_no_undefined_identifiers(
    ast,
    *,
    declared_variables: list[str] | None,
    line_num: int | None = None,
):
    allowed_vars = set(declared_variables or [])

    allowed_vars.add("ans")

    allowed_constants = {"epsilon", "pi", "euler"}
    allowed_ops = set(INFIX_OPERATIONS) | set(OPERATION_EXPANSIONS.keys())
    allowed_funcs = set(math_funcs.keys())

    def walk(node):
        if isinstance(node, list):
            if node and isinstance(node[0], str):
                head = node[0]
                if (
                    head.isidentifier()
                    and head not in allowed_ops
                    and head not in allowed_funcs
                ):
                    raise SyntaxError(
                        _format_error(
                            f"Malformed expression: {generate_expression(node)}",
                            line_num,
                        )
                    ) from None
            for child in node[1:] if (node and isinstance(node[0], str)) else node:
                walk(child)
            return

        if isinstance(node, (int, float)):
            return

        if isinstance(node, str):
            if node in allowed_constants:
                return
            if _is_number_like(node):
                return

            if node in allowed_ops:
                return

            if node.isidentifier():
                if node not in allowed_vars:
                    raise SyntaxError(
                        _format_error(f"Undefined identifier: {node}", line_num)
                    ) from None
                return

    walk(ast)


def substitute_vars(tokens, subst_vars):
    if isinstance(tokens, list):
        return [substitute_vars(token, subst_vars) for token in tokens]
    elif tokens in subst_vars:
        return subst_vars[tokens]
    else:
        return tokens


def parse_expression(tokens, subst_vars, functions, line_num):
    _validate_balanced_parentheses(tokens, line_num=line_num)
    print("Parsing expression:", tokens)
    result = substitute_vars(tokens, subst_vars)
    print("After substitution:", result)
    try:
        ast = generate_ast(result)
    except SyntaxError as e:
        raise SyntaxError(_format_error(str(e), line_num)) from None
    print("Generated AST:", ast)

    if len(ast) == 0:
        raise SyntaxError(_format_error("Empty expression", line_num))

    try:
        ast = infix_to_postfix(ast)
    except IndexError:
        raise SyntaxError(_format_error("Malformed expression", line_num)) from None
    except SyntaxError as e:
        raise SyntaxError(_format_error(str(e), line_num)) from None

    print("Postfix AST:", ast)
    expr = expand_expression(ast)
    print("Expanded AST:", expr)
    try:
        expr = expand_functions(expr, functions)
    except Exception as e:
        raise type(e)(_format_error(str(e), line_num)) from None
    print("After function expansion:", expr)

    expr = normalize_ast(expr)
    print("Normalized AST:", expr)
    print("")
    return expr


def parse_function(
    func_tokens,
    lines,
    subst_vars,
    functions,
    name,
    params=None,
    declared_variables=None,
    *,
    line_nums: list[int] | None = None,
    header_line_num: int | None = None,
):
    local_scope = subst_vars.copy()
    for line_no, tokens in enumerate(func_tokens):
        statement_line_num = (
            line_nums[line_no]
            if (line_nums is not None and line_no < len(line_nums))
            else None
        )
        if len(tokens) > 1 and tokens[1] == "=":
            _raise_if_conflicts_with_declared_variables(
                tokens[0],
                declared_variables,
                kind="Substitution variable",
                line_num=statement_line_num,
            )
            local_scope[tokens[0]] = parse_expression(
                tokens[2:], local_scope, functions, statement_line_num
            )
        elif len(tokens) > 1 and tokens[1] == ":=":
            var_name = tokens[0]
            _raise_if_conflicts_with_declared_variables(
                var_name,
                declared_variables,
                kind="Substitution variable",
                line_num=statement_line_num,
            )
            expr = lines[line_no].split(":=", 1)[1].strip()
            eval_namespace = {
                "__builtins__": builtins.__dict__,
                "vars": EvalVars(local_scope),
                **math_funcs,
            }
            try:
                result = eval(expr, eval_namespace)
            except Exception as e:
                raise ValueError(
                    _format_error(f"Python eval failed: {e}", statement_line_num)
                ) from None
            local_scope[var_name] = [str(result)]

        elif tokens and tokens[0] == "return":
            expr_tokens = tokens[1:]
            expr = parse_expression(
                expr_tokens, local_scope, functions, statement_line_num
            )
            expr = fix_structure(expr)
            _validate_no_undefined_identifiers(
                expr,
                declared_variables=(declared_variables or []) + (params or []),
                line_num=statement_line_num,
            )
            return expr

        else:
            raise SyntaxError(
                _format_error(
                    f"Unknown statement in function \"{name}\": {' '.join(tokens)}",
                    statement_line_num,
                )
            )

    # no return statement
    raise SyntaxError(
        _format_error(f'Function "{name}" missing return statement', header_line_num)
    )


def parse_repeat(
    body_tokens,
    lines,
    times,
    repeat_var,
    subst_vars,
    variables,
    functions,
    config,
    *,
    outer_bindings=None,
    output_state=None,
    line_nums: list[int] | None = None,
    header_line_num: int | None = None,
):
    if outer_bindings is None:
        outer_bindings = {}

    if output_state is None:
        output_state = {"output": None}

    variables = variables or []

    if repeat_var is not None:
        _raise_if_conflicts_with_declared_variables(
            repeat_var,
            variables,
            kind="Iteration variable",
            line_num=header_line_num,
        )
        if repeat_var in subst_vars.keys() or repeat_var in outer_bindings:
            raise ValueError(
                _format_error(
                    f"Iteration variable {repeat_var!r} conflicts with an existing substitution or loop binding",
                    header_line_num,
                )
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
    try:
        total_iterations = num(times)
    except Exception as e:
        raise ValueError(_format_error(str(e), header_line_num)) from None

    for itr in range(total_iterations):
        itr_bindings = (
            outer_bindings
            if repeat_var is None
            else outer_bindings | {repeat_var: str(itr)}
        )

        line_no = 0
        while line_no < len(body_tokens):
            tokens = strip_all_tabs(body_tokens[line_no])
            statement_line_num = (
                line_nums[line_no]
                if (line_nums is not None and line_no < len(line_nums))
                else None
            )
            if not tokens:
                line_no += 1
                continue

            if len(tokens) > 1 and tokens[1] == "=":
                _raise_if_conflicts_with_declared_variables(
                    tokens[0],
                    variables,
                    kind="Substitution variable",
                    line_num=statement_line_num,
                )
                subst_vars[tokens[0]] = parse_expression(
                    tokens[2:], subst_vars | itr_bindings, functions, statement_line_num
                )
                line_no += 1
                continue

            if len(tokens) > 1 and tokens[1] == ":=":
                var_name = tokens[0]
                _raise_if_conflicts_with_declared_variables(
                    var_name,
                    variables,
                    kind="Substitution variable",
                    line_num=statement_line_num,
                )
                expr = lines[line_no].split(":=", 1)[1].strip()
                eval_namespace = {
                    "__builtins__": builtins.__dict__,
                    "vars": EvalVars(subst_vars | itr_bindings),
                    **math_funcs,
                }
                try:
                    result = eval(expr, eval_namespace)
                except Exception as e:
                    raise ValueError(
                        _format_error(f"Python eval failed: {e}", statement_line_num)
                    ) from None
                subst_vars[var_name] = [str(result)]
                line_no += 1
                continue

            if tokens[0] == "repeat":
                if len(tokens) < 2:
                    raise SyntaxError(
                        _format_error(
                            "Repeat requires at least a repetition count",
                            statement_line_num,
                        )
                    )

                nested_times = tokens[1]
                nested_repeat_var = tokens[2] if len(tokens) >= 3 else None

                nested_body_tokens = []
                nested_body_lines = []
                nested_line_nums = []
                next_line_no = line_no + 1

                while next_line_no < len(body_tokens) and lines[
                    next_line_no
                ].startswith("    "):
                    nested_body_lines.append(lines[next_line_no][4:])
                    nested_body_tokens.append(dedent_once(body_tokens[next_line_no]))
                    if line_nums is not None and next_line_no < len(line_nums):
                        nested_line_nums.append(line_nums[next_line_no])
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
                        config,
                        outer_bindings=itr_bindings,
                        output_state=output_state,
                        line_nums=nested_line_nums,
                        header_line_num=statement_line_num,
                    )
                )

                line_no = next_line_no
                continue

            if tokens[0] == "out":
                if len(tokens) < 2:
                    raise SyntaxError(
                        _format_error(
                            "Out requires a variable name", statement_line_num
                        )
                    )
                output_var = tokens[1]
                if output_var in variables:
                    output_state["output"] = output_var
                else:
                    raise ValueError(
                        _format_error(
                            f"Output variable {output_var!r} not declared in variables",
                            statement_line_num,
                        )
                    )
                line_no += 1
                continue

            if tokens[0] == "return":
                expr_tokens = tokens[1:]
                expr = parse_expression(
                    expr_tokens,
                    subst_vars | itr_bindings,
                    functions,
                    statement_line_num,
                )
                expr = fix_structure(expr)

                _validate_no_undefined_identifiers(
                    expr,
                    declared_variables=variables,
                    line_num=statement_line_num,
                )
                results.append(
                    Output(
                        output=output_state["output"],
                        ast=expr,
                        config=replace(config, output=output_state["output"]),
                    )
                )
                output_state["output"] = None
                line_no += 1
                continue

            raise SyntaxError(
                _format_error(
                    f"Unknown statement in repeat: {' '.join(tokens)}",
                    statement_line_num,
                )
            )

    return results


def parse_pm3_to_ast(code):
    config = ProgramConfig()
    subst_vars = {}

    output_state = {"output": None}

    collecting_function = False
    collecting_loop = False

    funcs = []
    compiled = []
    for line_num, line in enumerate(code, start=1):
        line = line.split("#", 1)[0].rstrip()
        if not line:
            continue

        tokens = tokenize(line)
        # print(line, tokens)

        if collecting_function:
            if not tokens[0] == "__TAB__":
                collecting_function = False
                current_function.body = parse_function(
                    function_tokens,
                    function_body,
                    subst_vars,
                    funcs,
                    function_name,
                    current_function.params,
                    config.variables,
                    line_nums=function_line_nums,
                    header_line_num=function_header_line_num,
                )
                funcs.append(current_function)
            else:
                function_body.append(line[4:])
                function_tokens.append(tokens[1:])
                function_line_nums.append(line_num)
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
                    config,
                    output_state=output_state,
                    line_nums=repeat_line_nums,
                    header_line_num=repeat_header_line_num,
                )
                compiled.extend(repeat_result)
                config.output = output_state["output"]
            else:
                repeat_body.append(line[4:])
                repeat_tokens.append(tokens[1:])
                repeat_line_nums.append(line_num)

        if not collecting_function and not collecting_loop:
            if tokens[0] == "__TAB__":
                raise ValueError(
                    _format_error(
                        "Unexpected indentation outside function or loop",
                        line_num,
                    )
                )

            if tokens[0] == "def":
                if collecting_loop:
                    raise ValueError(
                        _format_error(
                            "Function definitions inside loops are not supported",
                            line_num,
                        )
                    )
                collecting_function = True
                function_body = []
                function_tokens = []
                function_line_nums = []
                function_name = tokens[1]
                params = tokens[2:]
                current_function = FunctionDef(
                    name=function_name, params=params, body=[]
                )
                function_header_line_num = line_num
            elif tokens[0] == "repeat":
                collecting_loop = True
                repeat_body = []
                repeat_tokens = []
                repeat_line_nums = []
                repeat_times = tokens[1]
                repeat_var = tokens[2] if len(tokens) >= 3 else None
                repeat_header_line_num = line_num

            elif tokens[0] == "out":
                output = tokens[1]
                if output in config.variables:
                    config.output = output
                    output_state["output"] = output
                else:
                    raise ValueError(
                        _format_error(
                            f"Output variable {output!r} not declared in variables",
                            line_num,
                        )
                    )

            elif tokens[0] == "return":
                if config.precision is None:
                    raise ValueError(
                        _format_error(
                            "Precision must be set before return statement",
                            line_num,
                        )
                    )
                if config.epsilon is None:
                    raise ValueError(
                        _format_error(
                            "Epsilon must be set before return statement",
                            line_num,
                        )
                    )
                expr_tokens = tokens[1:]
                expr = parse_expression(expr_tokens, subst_vars, funcs, line_num)
                expr = fix_structure(expr)

                _validate_no_undefined_identifiers(
                    expr,
                    declared_variables=config.variables,
                    line_num=line_num,
                )
                compiled.append(
                    Output(
                        output=output_state["output"],
                        ast=expr,
                        config=replace(config, output=output_state["output"]),
                    )
                )
                config.output = None
                output_state["output"] = None

            elif tokens[0] == "//":
                if not collecting_function and not collecting_loop:
                    try:
                        parse_pragma(line.strip()[2:].strip(), config)
                    except Exception as e:
                        raise type(e)(_format_error(str(e), line_num)) from None
                else:
                    raise ValueError(
                        _format_error(
                            "Pragmas inside functions or loops are not supported",
                            line_num,
                        )
                    )

            else:
                # non-function, non-pragma line
                if len(tokens) > 1 and tokens[1] == "=":
                    _raise_if_conflicts_with_declared_variables(
                        tokens[0],
                        config.variables,
                        kind="Substitution variable",
                        line_num=line_num,
                    )
                    subst_vars[tokens[0]] = tokens[2:]
                elif len(tokens) > 1 and tokens[1] == ":=":
                    # python eval!
                    var_name = tokens[0]
                    _raise_if_conflicts_with_declared_variables(
                        var_name,
                        config.variables,
                        kind="Substitution variable",
                        line_num=line_num,
                    )
                    expr = line.split(":=")[1].strip()
                    eval_namespace = {
                        "__builtins__": builtins.__dict__,
                        "vars": EvalVars(subst_vars),
                        **math_funcs,
                    }
                    try:
                        result = eval(expr, eval_namespace)
                    except Exception as e:
                        raise ValueError(
                            _format_error(f"Python eval failed: {e}", line_num)
                        ) from None
                    subst_vars[var_name] = [str(result)]
                else:
                    raise SyntaxError(
                        _format_error(
                            f"Unknown statement: {' '.join(tokens)}",
                            line_num,
                        )
                    )

    if collecting_function:
        current_function.body = parse_function(
            function_tokens,
            function_body,
            subst_vars,
            funcs,
            function_name,
            current_function.params,
            config.variables,
            line_nums=function_line_nums,
            header_line_num=function_header_line_num,
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
            config,
            output_state=output_state,
            line_nums=repeat_line_nums,
            header_line_num=repeat_header_line_num,
        )
        compiled.extend(repeat_result)
        config.output = output_state["output"]

    # print(config)
    # print(subst_vars)
    # print(funcs)

    output = []
    for i, item in enumerate(compiled):
        output.append(fix_structure(item))

    return compiled, config


def generate_expression(ast):
    if isinstance(ast, list):
        if not ast:
            return ""
        if isinstance(ast[0], str):
            func_name = ast[0]
            if func_name in INFIX_OPERATIONS and len(ast) == 3:
                left = generate_expression(ast[1])
                right = generate_expression(ast[2])
                return f"({left} {func_name} {right})"
            args = [generate_expression(arg) for arg in ast[1:]]
            return f"{func_name}({', '.join(args)})"
        else:
            return "(" + " ".join(generate_expression(elem) for elem in ast) + ")"
    else:
        return str(ast)


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
            if config.precision is not None and isinstance(result, float):
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


def expr_length(ast) -> int:
    return len(generate_expression(ast).replace(" ", ""))


def find_beneficial_duplicates(ast, min_length: int = 4, min_savings: int = -999):
    seen = {}

    if not isinstance(ast, list):
        return []

    for sub in all_sublists(ast):
        key = json.dumps(sub, sort_keys=False, default=str)
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


if __name__ == "__main__":
    with open("mockup.pm3") as f:
        code = [line.rstrip() for line in f.readlines() if line]
        asts, config = parse_pm3_to_ast(code)
        outputs = []
        for i, output in enumerate(asts):
            if output.config.simplify:
                output.ast = simplify_ast(output.ast, output.config)
                line = AstOutput(output=output.output, ast=output.ast)

            else:
                line = AstOutput(output=output.output, ast=output.ast)

            if output.config.dupe:
                for extraction, variable in extract_subexpressions(
                    line.ast,
                    line.output,
                    min_savings=output.config.dupe_min_savings,
                ):
                    outputs.append(AstOutput(output=variable, ast=extraction))
            else:
                outputs.append(line)

        with open("math.txt", "w") as f:
            for i, output in enumerate(outputs):
                expr = generate_expression(output.ast)
                f.write(f"to {output.output}:\n")
                f.write(expr + "\n")
                print(f"to {output.output}:")
                print(expr)
