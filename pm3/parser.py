from __future__ import annotations

import builtins
from dataclasses import replace

from .ast_gen import generate_ast, infix_to_postfix
from .ast_transform import expand_expression, expand_functions, normalize_ast
from .ast_utils import fix_structure, substitute_vars
from .config import Output
from .constants import math_funcs
from .logging import _preview_sequence, print_debug, print_verbose
from .utils import EvalVars, num
from .validation import (
    _raise_if_conflicts_with_declared_variables,
    _validate_balanced_parentheses,
    _validate_no_undefined_identifiers,
)


def parse_expression(tokens, subst_vars, functions, line_num, progress_cb=None):
    _validate_balanced_parentheses(tokens, line_num=line_num)

    if line_num is not None:
        print_verbose(f"line {line_num}: parsing expression")
    print_debug(
        f"parse_expression: raw tokens={_preview_sequence(tokens, max_items=32)}"
    )

    if progress_cb is not None:
        try:
            progress_cb(None, "substituting vars")
        except Exception:
            pass
    result = substitute_vars(tokens, subst_vars)

    print_debug(
        "parse_expression: after substitution tokens="
        + _preview_sequence(result, max_items=32)
    )
    print_verbose("After substitution: " + str(result))

    try:
        if progress_cb is not None:
            try:
                progress_cb(None, "generating ast")
            except Exception:
                pass
        ast = generate_ast(result)
    except SyntaxError as e:
        msg = str(e).strip()
        if line_num is not None and not msg.lower().startswith("line "):
            msg = f"Line {line_num}: {msg}"
        raise SyntaxError(msg) from None

    print_verbose("Generated AST: " + str(ast))

    if len(ast) == 0:
        msg = "Empty expression"
        if line_num is not None:
            msg = f"Line {line_num}: {msg}"
        raise SyntaxError(msg)

    try:
        if progress_cb is not None:
            try:
                progress_cb(None, "infix to postfix")
            except Exception:
                pass
        ast = infix_to_postfix(ast)
    except IndexError:
        msg = "Malformed expression"
        if line_num is not None:
            msg = f"Line {line_num}: {msg}"
        raise SyntaxError(msg) from None
    except SyntaxError as e:
        msg = str(e).strip()
        if line_num is not None and not msg.lower().startswith("line "):
            msg = f"Line {line_num}: {msg}"
        raise SyntaxError(msg) from None

    print_debug(f"parse_expression: postfix ast type={type(ast).__name__}")
    print_verbose("Postfix AST: " + str(ast))

    if progress_cb is not None:
        try:
            progress_cb(None, "expanding expression")
        except Exception:
            pass
    expr = expand_expression(ast)
    print_debug(f"parse_expression: expanded ast type={type(expr).__name__}")
    print_verbose("Expanded AST: " + str(expr))

    try:
        if progress_cb is not None:
            try:
                progress_cb(None, "expanding functions")
            except Exception:
                pass
        expr = expand_functions(expr, functions)
    except Exception as e:
        msg = str(e).strip()
        if line_num is not None and not msg.lower().startswith("line "):
            msg = f"Line {line_num}: {msg}"
        raise type(e)(msg) from None

    print_verbose("After function expansion: " + str(expr))

    if progress_cb is not None:
        try:
            progress_cb(None, "normalizing")
        except Exception:
            pass
    expr = normalize_ast(expr)
    print_debug(f"parse_expression: normalized ast type={type(expr).__name__}")
    print_verbose("Normalized AST: " + str(expr))
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
    progress_cb=None,
):
    local_scope = subst_vars.copy()
    for line_no, tokens in enumerate(func_tokens):
        if progress_cb is not None:
            try:
                progress_cb(
                    None, f"function body line {line_no + 1}/{len(func_tokens)}"
                )
            except Exception:
                pass

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
                tokens[2:],
                local_scope,
                functions,
                statement_line_num,
                progress_cb=progress_cb,
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
                msg = f"Python eval failed: {e}"
                if statement_line_num is not None:
                    msg = f"Line {statement_line_num}: {msg}"
                raise ValueError(msg) from None
            local_scope[var_name] = [str(result)]

        elif tokens and tokens[0] == "return":
            expr_tokens = tokens[1:]
            expr = parse_expression(
                expr_tokens,
                local_scope,
                functions,
                statement_line_num,
                progress_cb=progress_cb,
            )
            expr = fix_structure(expr)
            _validate_no_undefined_identifiers(
                expr,
                declared_variables=(declared_variables or []) + (params or []),
                line_num=statement_line_num,
            )
            return expr

        else:
            msg = f"Unknown statement in function \"{name}\": {' '.join(tokens)}"
            if statement_line_num is not None:
                msg = f"Line {statement_line_num}: {msg}"
            raise SyntaxError(msg)

    msg = f'Function "{name}" missing return statement'
    if header_line_num is not None:
        msg = f"Line {header_line_num}: {msg}"
    raise SyntaxError(msg)


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
    progress_cb=None,
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
            msg = f"Iteration variable {repeat_var!r} conflicts with an existing substitution or loop binding"
            if header_line_num is not None:
                msg = f"Line {header_line_num}: {msg}"
            raise ValueError(msg)

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
        msg = str(e).strip()
        if header_line_num is not None and not msg.lower().startswith("line "):
            msg = f"Line {header_line_num}: {msg}"
        raise ValueError(msg) from None

    for itr in range(total_iterations):
        if progress_cb is not None:
            try:
                progress_cb(None, f"iteration {itr + 1}/{total_iterations}")
            except Exception:
                pass

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
                    tokens[2:],
                    subst_vars | itr_bindings,
                    functions,
                    statement_line_num,
                    progress_cb=progress_cb,
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
                    msg = f"Python eval failed: {e}"
                    if statement_line_num is not None:
                        msg = f"Line {statement_line_num}: {msg}"
                    raise ValueError(msg) from None
                subst_vars[var_name] = [str(result)]
                line_no += 1
                continue

            if tokens[0] == "repeat":
                if len(tokens) < 2:
                    msg = "Repeat requires at least a repetition count"
                    if statement_line_num is not None:
                        msg = f"Line {statement_line_num}: {msg}"
                    raise SyntaxError(msg)

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
                        progress_cb=progress_cb,
                    )
                )

                line_no = next_line_no
                continue

            if tokens[0] == "out":
                if len(tokens) < 2:
                    msg = "Out requires a variable name"
                    if statement_line_num is not None:
                        msg = f"Line {statement_line_num}: {msg}"
                    raise SyntaxError(msg)
                output_var = tokens[1]
                if output_var in variables:
                    output_state["output"] = output_var
                else:
                    msg = f"Output variable {output_var!r} not declared in variables"
                    if statement_line_num is not None:
                        msg = f"Line {statement_line_num}: {msg}"
                    raise ValueError(msg)
                line_no += 1
                continue

            if tokens[0] == "return":
                expr_tokens = tokens[1:]
                expr = parse_expression(
                    expr_tokens,
                    subst_vars | itr_bindings,
                    functions,
                    statement_line_num,
                    progress_cb=progress_cb,
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
                (
                    f"Line {statement_line_num}: Unknown statement in repeat: {' '.join(tokens)}"
                    if statement_line_num is not None
                    else f"Unknown statement in repeat: {' '.join(tokens)}"
                )
            )

    return results
