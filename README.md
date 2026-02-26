# The Paramath Programming Language, ver 3.

Paramath is a Domain-Specific Language (DSL) that transforms procedural math code into mathematical expressions. Paramath compiles into math expressions with operations commonly found on standard scientific calculators, allowing evaluation of logical code on mathematical hardware.

All this, powered by Python.

> **Why the name "Paramath"?**  
> The word "paramath" comes from the portmanteau of "parenthesis" and "mathematics".

## Features

- **Infix + prefix syntax**: Arithmetic and comparators are infix; other operations are function-style prefix calls
- **Automatic optimization**: Simplification and duplicate subexpression extraction
- **Loop unrolling**: Compile-time iteration for performance
- **Flexible output**: Display and/or write compiled expressions to files

## Documentation

- Language reference: [paramath_syntax_guide.md](https://github.com/kaemori/paramath3/blob/main/docs/paramath_syntax_guide.md)
- Developer docs: [paramath_docs_developers.md](https://github.com/kaemori/paramath3/blob/main/docs/paramath_docs_developers.md)

## Installation

### Method 1 (PyPI)

#### Generic PyPI install (virtual environment)

```bash
python -m venv .venv
source .venv/bin/activate
pip install paramath3
```

Run the compiler:

```bash
paramath3 --version
```

#### Recommended on Arch Linux: use pipx

Arch enforces Python's externally managed environment (PEP 668), so global `pip install` may fail. Use `pipx` instead:

```bash
sudo pacman -S --needed python-pipx
pipx ensurepath
```

Restart your shell, then install Paramath:

```bash
pipx install paramath3[full]
```

To install without optional dependencies:

```bash
pipx install paramath3
```

To add optional dependencies to an existing pipx installation:

```bash
pipx inject paramath3 sympy tqdm
```

### Method 2 (Direct Git Clone)

Clone the repository:

```bash
git clone https://github.com/kaemori/paramath3.git
cd paramath3
```

Run directly from source:

```bash
python cli.py --version
```

Install as a local package (recommended):

```bash
pip install -e .
paramath3 --version
```

Optional dependencies:

```bash
pip install -e .[full]
```

## Usage

The compiler can be run with:

```bash
python cli.py [-h] [-V] [-o FILE] [-d] [-v] [-s] [-S] [-O] [-m] [-t] [-L FILE] [-p] [--no-color] [filepath]
```

Options:

- `filepath` input Paramath file
- `-h, --help` show help and exit
- `-V, --version` print Paramath version and exit
- `-o, --output FILE` output file (default: `math.txt`)
- `-d, --debug` enable debug output
- `-v, --verbose` enable verbose output
- `-s, --suppress` suppress non-error output
- `-S, --silent` suppress all output
- `-O, --print-output` print compiled output
- `-m, --math-output` format output for calculators (`^`, implicit multiplication, `ANS`)
- `-t, --trust` allow compile-time Python eval (`:=`)
- `-L, --logfile FILE` write logs to file
- `-p, --progress` show parsing/processing progress bars
- `--no-color` disable colored CLI output

## Testing

Test layout:

- `paramath_test/` contains `.pm3` fixture programs
- `tests/` contains Python test modules (`.py`)

Run the Paramath v3 suite:

```bash
python -m unittest tests/pm_test_suite.py -v
```

The fixture set in `paramath_test/` is a file-for-file (`1:1`) conversion of `_for_ref/paramath/paramath_tests` into Paramath v3 syntax, including `examples/number_test`.

## License

This project is licensed under the [MIT License](https://github.com/kaemori/paramath3/blob/main/LICENSE).
