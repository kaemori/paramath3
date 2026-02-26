# Paramath v3 — Developer Documentation

This document describes the current code in this repository (v3.1.3).

## Overview

- Paramath is a small DSL compiler for compact mathematical programs.
- The core implementation is currently in a single module: `paramath.py`.
- The CLI entrypoint is `cli.py`.

## Installation

Required: Python 3.10+ (recommended).

Optional dependencies:

- `tqdm` for progress bars (`-p/--progress`)
- `sympy` for SymPy simplification (`//sympy true`)

```bash
pip install tqdm sympy
```

## CLI (`cli.py`)

Examples:

```bash
python cli.py test.pm3
python cli.py test.pm3 -dv
python cli.py test.pm3 -t -m -O -o compiled.txt
```

Arguments and flags:

- positional: `filepath` (input script)
- `-V, --version` show version and exit
- `-o, --output FILE` output file (default: `math.txt`)
- `-d, --debug` enable debug output
- `-v, --verbose` enable verbose output
- `-s, --suppress` suppress non-error output
- `-S, --silent` suppress all output
- `-O, --print-output` print compiled expressions to stdout
- `-m, --math-output` math-calculator formatting (`** -> ^`, remove `*`, `ans -> ANS`, `pi -> π`, `e -> ℯ`)
- `-t, --trust` allow compile-time Python eval via `:=`
- `-L, --logfile FILE` write debug/verbose logs to a file
- `-p, --progress` show progress bars (requires `tqdm`)
- `--no-color` disable ANSI color output

Important runtime behavior:

- Without `-t/--trust`, any script containing `:=` is rejected after listing offending lines.
- `-p/--progress` cannot be used together with `-d/--debug` or `-v/--verbose`.

## Public Programmatic API (`paramath.py`)

Primary functions:

- `code_to_lines(source)`
    - Normalizes source text/lines.
    - Splits semicolon-separated statements.
    - Preserves parenthesized multiline expressions.

- `parse_pm3_to_ast(code, *, progress: bool = False)`
    - Parses normalized lines into compiler outputs.
    - Returns a list of `Output` objects (not a `(asts, config)` tuple).

- `process_asts(asts, *, progress: bool = False)`
    - Takes list of `Output` objects from `parse_pm3_to_ast()`.
    - Returns list of `(output_name, expression_string)` tuples.

- `check_python_eval(code)`
    - Returns `[(line_number, line_text), ...]` for lines containing `:=`.

Utilities used by CLI and integration code:

- `file_to_lines(filename)`
- `print_debug(message)`
- `print_verbose(message)`

Data classes:

- `ProgramConfig` compiler settings (`precision`, `epsilon`, `variables`, `dupe`, `simplify`, `sympy`, etc.)
- `Output` parsed output payload (`output`, `ast`, `config`)
- `FunctionDef`, `AstOutput`

## Compiler Flow (Current Implementation)

1. Load source (`cli.py`) and normalize it with `code_to_lines()`.
2. Parse with `parse_pm3_to_ast()`:
    - tokenization (`tokenize`)
    - AST construction (`generate_ast` + `infix_to_postfix`)
    - pragma handling (`parse_pragma`)
    - substitution/function/repeat parsing
    - output finalization (`_finalize_output_line_ast`)
3. Process with `process_asts()`:
    - optional simplification
    - optional duplicate extraction
    - expression generation (`generate_expression`)
    - optional SymPy pass (single/multicore helper)

## Pragmas Supported

- `//precision <int>`
- `//epsilon <float>`
- `//variables <space-separated names>`
- `//dupe <true|false> [min_savings:int]`
- `//simplify <true|false>`
- `//sympy <true|false>`

Notes:

- `precision` and `epsilon` must be set before any `return` statement.
- `//sympy true` requires `sympy` installed.

## Programmatic Example

```python
from paramath import code_to_lines, parse_pm3_to_ast, process_asts, check_python_eval

source = """
//precision 6
//epsilon 1e-9
//variables x y z
x = 2
return x + 3
"""

code = code_to_lines(source)
unsafe_lines = check_python_eval(code)
if unsafe_lines:
    raise RuntimeError("Untrusted source contains := lines")

asts = parse_pm3_to_ast(code)
outputs = process_asts(asts)

for output_name, expr in outputs:
    print(f"to {output_name}:")
    print(expr)
```

## Repository Layout (Current)

- `cli.py` CLI entrypoint
- `paramath.py` core parser/compiler/simplifier/codegen and public API
- `colors.py` ANSI color helpers for CLI output
- `github.com/kaemori/paramath3/docs/paramath_syntax_guide.md` language reference
- `github.com/kaemori/paramath3/docs/paramath_docs_developers.md` this file

## Validation / Quick Checks

- CLI smoke test:

```bash
python cli.py test.pm3 -O
```

- Progress mode check:

```bash
python cli.py test.pm3 -p
```

- SymPy check:

```bash
python cli.py test.pm3 -O
```

with `//sympy true` in the input script.
