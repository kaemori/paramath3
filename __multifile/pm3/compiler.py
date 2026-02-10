from __future__ import annotations

import builtins
from dataclasses import replace

from .ast_utils import fix_structure
from .codegen import generate_expression
from .config import AstOutput, FunctionDef, Output, ProgramConfig
from .dedupe import extract_subexpressions
from .logging import print_verbose
from .parser import parse_expression, parse_function, parse_repeat
from .pragma import parse_pragma
from .simplify import simplify_ast
from .sympy_utils import simplify_sympy_multicore
from .tokenizer import tokenize
from .utils import EvalVars
from .validation import (
    _raise_if_conflicts_with_declared_variables,
    _validate_no_undefined_identifiers,
)


def parse_pm3_to_ast(code, *, progress: bool = False):
    print_verbose(f"starting parse_pm3_to_ast: {len(code)} lines")
    config = ProgramConfig()
    subst_vars = {}

    output_state = {"output": None}

    collecting_function = False
    collecting_loop = False

    funcs: list[FunctionDef] = []
    compiled: list[Output] = []

    iterator = enumerate(code, start=1)
    bar = None
    if progress:
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            tqdm = None

        total = len(code) if hasattr(code, "__len__") else None
        if tqdm is not None:
            bar = tqdm(iterator, total=total, desc="pm3", unit="line")
            iterator = bar

    current_stage = "scanning"
    current_substage = None
    current_preview = None

    def _render_postfix() -> str:
        parts = []
        if current_substage:
            parts.append(str(current_substage))
        if current_preview:
            parts.append(str(current_preview))
        return " | ".join(parts)

    def progress_update(stage=None, substage=None):
        nonlocal current_stage, current_substage
        if bar is None:
            return

        changed = False
        if stage is not None and stage != current_stage:
            current_stage = stage
            try:
                bar.set_description_str(f"pm3 - {current_stage}")
            except Exception:
                pass
            changed = True

        if substage is not None and substage != current_substage:
            current_substage = substage
            changed = True

        if changed:
            try:
                bar.set_postfix_str(_render_postfix())
            except Exception:
                pass

    for line_num, line in iterator:
        if bar is not None:
            try:
                preview = line.split("#", 1)[0].strip()
                if len(preview) > 60:
                    preview = preview[:57] + "..."
                current_preview = f"L{line_num}: {preview}"
                bar.set_postfix_str(_render_postfix())
            except Exception:
                pass

        line = line.split("#", 1)[0].rstrip()
        if not line:
            continue

        tokens = tokenize(line)

        if collecting_function:
            if not tokens[0] == "__TAB__":
                collecting_function = False
                progress_update(f"parsing function {function_name}", "parsing body")
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
                    progress_cb=progress_update,
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
                progress_update("unwrapping repeat", "parsing body")
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
                    progress_cb=progress_update,
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
                    f"Line {line_num}: Unexpected indentation outside function or loop"
                )

            if tokens[0] == "def":
                collecting_function = True
                function_body = []
                function_tokens = []
                function_line_nums = []
                function_name = tokens[1]
                progress_update(f"collecting function {function_name}", "reading body")
                params = tokens[2:]
                current_function = FunctionDef(
                    name=function_name,
                    params=params,
                    body=[],
                )
                function_header_line_num = line_num

            elif tokens[0] == "repeat":
                collecting_loop = True
                progress_update("collecting repeat", "reading body")
                repeat_body = []
                repeat_tokens = []
                repeat_line_nums = []
                repeat_times = tokens[1]
                repeat_var = tokens[2] if len(tokens) >= 3 else None
                repeat_header_line_num = line_num

            elif tokens[0] == "out":
                progress_update("parsing main body", "output selection")
                output = tokens[1]
                if output in config.variables:
                    config.output = output
                    output_state["output"] = output
                else:
                    raise ValueError(
                        f"Line {line_num}: Output variable {output!r} not declared in variables"
                    )

            elif tokens[0] == "return":
                progress_update("parsing main body", "parsing return")
                if config.precision is None:
                    raise ValueError(
                        f"Line {line_num}: Precision must be set before return statement"
                    )
                if config.epsilon is None:
                    raise ValueError(
                        f"Line {line_num}: Epsilon must be set before return statement"
                    )

                expr_tokens = tokens[1:]
                expr = parse_expression(
                    expr_tokens,
                    subst_vars,
                    funcs,
                    line_num,
                    progress_cb=progress_update,
                )
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
                progress_update("parsing config", "pragma")
                try:
                    parse_pragma(line.strip()[2:].strip(), config)
                except Exception as e:
                    msg = str(e).strip()
                    if not msg.lower().startswith("line "):
                        msg = f"Line {line_num}: {msg}"
                    raise type(e)(msg) from None

            else:
                if len(tokens) > 1 and tokens[1] == "=":
                    progress_update("parsing main body", "substitution")
                    _raise_if_conflicts_with_declared_variables(
                        tokens[0],
                        config.variables,
                        kind="Substitution variable",
                        line_num=line_num,
                    )
                    subst_vars[tokens[0]] = tokens[2:]

                elif len(tokens) > 1 and tokens[1] == ":=":
                    progress_update("parsing main body", "python eval")
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
                    }
                    from .constants import math_funcs

                    eval_namespace |= math_funcs
                    try:
                        result = eval(expr, eval_namespace)
                    except Exception as e:
                        raise ValueError(
                            f"Line {line_num}: Python eval failed: {e}"
                        ) from None
                    subst_vars[var_name] = [str(result)]

                else:
                    raise SyntaxError(
                        f"Line {line_num}: Unknown statement: {' '.join(tokens)}"
                    )

    if collecting_function:
        progress_update(f"parsing function {function_name}", "parsing body")
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
            progress_cb=progress_update,
        )
        funcs.append(current_function)

    if collecting_loop:
        progress_update("unwrapping repeat", "parsing body")
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
            progress_cb=progress_update,
        )
        compiled.extend(repeat_result)
        config.output = output_state["output"]

    if bar is not None:
        try:
            bar.set_postfix_str("done")
            bar.close()
        except Exception:
            pass

    return compiled, config


def process_asts(asts, *, progress: bool = False):
    print_verbose(f"processing {len(asts)} AST output(s)")
    bar = None
    iterator = enumerate(asts)
    if progress:
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            tqdm = None
        if tqdm is not None:
            bar = tqdm(total=len(asts), desc="pm3 - processing", unit="ast")

    def progress_stage(stage: str | None = None):
        if bar is None:
            return
        if stage is None:
            return
        try:
            bar.set_postfix_str(stage)
        except Exception:
            pass

    try:
        outputs_ast: list[AstOutput] = []
        for _, output in iterator:
            progress_stage("building outputs")

            if output.config.simplify:
                progress_stage("simplifying")
                output.ast = simplify_ast(output.ast, output.config)
                line = AstOutput(
                    output=output.output,
                    ast=output.ast,
                    sympy=output.config.sympy,
                )
            else:
                line = AstOutput(
                    output=output.output,
                    ast=output.ast,
                    sympy=output.config.sympy,
                )

            if output.config.dupe:
                progress_stage("deduping")
                for extraction, variable in extract_subexpressions(
                    line.ast,
                    line.output,
                    min_savings=output.config.dupe_min_savings,
                ):
                    outputs_ast.append(
                        AstOutput(
                            output=variable,
                            ast=extraction,
                            sympy=output.config.sympy,
                        )
                    )
            else:
                outputs_ast.append(line)

            if bar is not None:
                try:
                    bar.update(1)
                except Exception:
                    pass

        exprs = [generate_expression(item.ast) for item in outputs_ast]

        if sum(item.sympy for item in outputs_ast):
            progress_stage("sympy")
            try:
                import sympy  # noqa: F401
            except ImportError:
                raise ImportError(
                    "SymPy is required for sympy output. Please install it via 'pip install sympy'."
                ) from None

            sympy_indexes = [i for i, item in enumerate(outputs_ast) if item.sympy]
            sympy_exprs = [exprs[i] for i in sympy_indexes]
            simplified_sympy_exprs = simplify_sympy_multicore(
                sympy_exprs,
                show_progress=progress,
            )
            for i, simplified in zip(sympy_indexes, simplified_sympy_exprs):
                exprs[i] = simplified

        outputs = [(item.output, exprs[i]) for i, item in enumerate(outputs_ast)]
        return outputs
    finally:
        if bar is not None:
            try:
                bar.set_postfix_str("done")
                bar.close()
            except Exception:
                pass


def check_python_eval(code):
    return [(i + 1, line) for i, line in enumerate(code) if ":=" in line]
