# The Paramath Programming Language

Paramath is a Domain-Specific Language (DSL) that transforms procedural math code into mathematical expressions. Paramath compiles into math expressions with operations commonly found on standard scientific calculators, allowing evaluation of logical code on mathematical hardware.

All this, powered by Python.

> **Why the name "Paramath"?**  
> The word "paramath" comes from the portmanteau of "parenthesis" and "mathematics".

## Features

- **Infix + prefix syntax**: Arithmetic and comparators are infix; other operations are function-style prefix calls
- **Automatic optimization**: Simplification and duplicate subexpression extraction
- **Loop unrolling**: Compile-time iteration for performance
- **Flexible output**: Display and/or write compiled expressions to files
- **Trust-aware execution**: Compile-time Python eval (`:=`) can be blocked unless trusted

## Documentation

- Language reference: [paramath_syntax_guide.md](paramath_syntax_guide.md)
- Developer docs: [paramath_docs_developers.md](paramath_docs_developers.md)

## Installation

### Method 1 (PyPI)

**TBD**

### Method 2 (Direct Git Clone)

Clone the repository:

```bash
git clone https://github.com/kaemori/paramath.git
cd paramath
```

Run directly from source:

```bash
python cli.py --version
```

Optional dependencies:

```bash
pip install tqdm sympy
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

## License

This project is licensed under the [MIT License](LICENSE).

TODO:
add pypi installation instructions once published
add unit tests and example scripts
