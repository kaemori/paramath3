# Paramath v3 — Documentation

This document describes Paramath v3 (code in this repository). It covers installation, CLI usage, the public programmatic API, key module responsibilities, and short examples.

**Overview**

- Paramath is a small domain-specific compiler for compact mathematical programs.
- The runtime core lives in the `pm3` package. The CLI entrypoint is `cli.py`.

**Quick install**

1. Clone the repository or place the source in your PYTHONPATH.
2. Optional dependencies (for extra features):
    - `sympy` (for symbolic simplification output),
    - `tqdm` (for progress bars).

Install optional packages if needed:

```bash
pip install sympy tqdm
```

No packaging is provided here; run tools directly from the repository or install via a simple `pip install -e .` if you create a `setup.py`/`pyproject.toml`.

**CLI — `cli.py`**

Usage examples (also shown in the program epilog):

```bash
# compile a file and write math.txt
paramath3 testfile.pm

# compile with verbose and debug logs
paramath3 testfile.pm -dv

# show output and write to compiled.txt
paramath3 testfile.pm -tmo compiled.txt -O
```

Common flags:

- `-o, --output FILE` : output file (default: `math.txt`).
- `-d, --debug` : enable debug logs.
- `-v, --verbose` : enable verbose logs.
- `-s, --suppress` : suppress non-error output.
- `-S, --silent` : suppress all output.
- `-O, --print-output` : print compiled output to stdout.
- `-m, --math-output` : format results for calculators (use `^`, implicit multiplication, `ANS`).
- `-t, --trust` : enable unsafe Python eval sections (`:=`) — use with caution.
- `-L, --logfile FILE` : path to logfile for verbose/debug output.
- `-p, --progress` : show parsing progress (requires `tqdm`).

**High-level programmatic API (public)**

The package re-exports the main public functions in `pm3/__init__.py`:

- `parse_pm3_to_ast(code, *, progress: bool = False)`
    - Input: `code` is an iterable of lines (list of strings). Parses a Paramath source into a list of `Output` entries (ASTs) and a `ProgramConfig`.
    - Returns: `(asts, config)` where `asts` is a list of outputs with ASTs and `config` is the aggregated configuration.

- `process_asts(asts, *, progress: bool = False)`
    - Input: `asts` as returned by `parse_pm3_to_ast`.
    - Performs simplification/dupe extraction/sympy simplification (if enabled) and returns compiled expressions suitable for textual output: a list of `(output_name, expression_string)` tuples.

- `check_python_eval(code)`
    - Scans `code` (iterable of lines) and returns a list of `(line_number, line)` pairs containing `':='` Python-eval lines; useful for trusting checks.

- Logging helpers exported: `print_debug`, `print_verbose` (see `pm3/logging.py`).

**Important modules and responsibilities**

- `pm3/tokenizer.py`
    - `tokenize(line: str) -> list[str]`: breaks a source line into tokens and handles indentation markers (tabs).

- `pm3/ast_gen.py`
    - `generate_ast(tokens)` : builds nested ASTs from token lists.
    - `infix_to_postfix(ast)` : converts ASTs with infix operators into explicit operator nodes (postfix-style AST for evaluation/codegen).

- `pm3/parser.py`
    - High-level parsing: `parse_expression`, `parse_function`, `parse_repeat` and orchestration in `parse_pm3_to_ast`.
    - Validates constructs, handles substitution variables and function bodies, and returns structured AST outputs.

- `pm3/compiler.py`
    - `process_asts(asts, progress=False)`: runs simplification, duplication extraction, sympy simplification (optional), and invokes code generation (`pm3.codegen.generate_expression`) to produce final string expressions.

- `pm3/codegen.py`
    - `generate_expression(ast)`: convert internal AST back into a readable string with proper parentheses and precedence.

- `pm3/simplify.py`
    - `simplify_ast(ast, config)`: performs literal folding and algebraic identity simplifications when requested by config.

- `pm3/utils.py`
    - Utilities such as `stable_serialize`, `num()`, and `EvalVars` used across the compiler.

**Program flow summary**

1. `cli.py` loads a file (list of lines) and calls `parse_pm3_to_ast(code, progress=...)`.
2. `parse_pm3_to_ast` tokenizes lines (`tokenize`), builds ASTs (`generate_ast`), converts infix to operator ASTs (`infix_to_postfix`), expands macros/functions, validates and collects outputs.
3. `process_asts` then simplifies, deduplicates subexpressions (if enabled), optionally runs SymPy simplifier, and uses `generate_expression` to produce final textual expressions.

**Using Paramath programmatically**

Minimal example (from Python):

```python
from pm3 import parse_pm3_to_ast, process_asts

code = [
    "//precision 6",
    "x = 2",
    "return x + 3",
]

asts, config = parse_pm3_to_ast(code)
outputs = process_asts(asts)
for output_name, expr in outputs:
    print(f"to {output_name}: {expr}")
```

If you have `:=`-style Python eval in your source, run `check_python_eval(code)` first and enforce a trust policy before allowing execution.

**Examples**

Given `example.pm`:

```
//precision 6
a = 2
return a * 3 + 1
```

Running via CLI:

```bash
python cli.py example.pm -O
```

Programmatic usage returns the compiled expression string via `process_asts`.

**Developer guide — repository layout**

- `cli.py` : CLI entrypoint and argument parsing.
- `pm3/` : core package. Key files:
    - `tokenizer.py`, `ast_gen.py`, `parser.py`, `compiler.py`, `codegen.py`, `simplify.py`, `utils.py`, `logging.py` etc.
- `colors.py` : CLI color helpers.
- `paramath_docs_v3.md` : this document.

When extending:

- Add parsing helpers to `tokenizer.py` / `ast_gen.py` if changing syntax.
- Keep high-level logic (validation, function/loop handling) in `parser.py`.
- Add transformation/simplification logic to `simplify.py` or new transform modules.
- Update `process_asts` in `compiler.py` to include new codegen/simplify steps.

**Testing and quick checks**

- There are no formal tests included. To validate, create small `.pm` source files and run `python cli.py path/to/file.pm`.
- For progress bars, install `tqdm` and run with `-p`.
- For SymPy-based simplification, install `sympy` and set `sympy` flags in your program config (via `//` pragmas).
