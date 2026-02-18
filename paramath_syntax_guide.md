# Paramath Language Reference

### v3.0.12 · A differentiable programming language for continuous computation

---

## Table of Contents

- [Introduction](#introduction)
- [Syntax Overview](#syntax-overview)
- [Data Types & Substitutions](#data-types--substitutions)
- [Operators & Functions](#operators--functions)
- [Program Structure](#program-structure)
- [Pragmas](#pragmas)
- [Function Definitions](#function-definitions)
- [Loops & Iteration](#loops--iteration)
- [Intermediates & Code Blocks](#intermediates--code-blocks)
- [Advanced Features](#advanced-features)
- [Complete Examples](#complete-examples)
- [Command-Line Interface](#command-line-interface)
- [Error Reference](#error-reference)
- [Best Practices](#best-practices)

---

## Introduction

**Paramath** is a domain-specific language designed for applications that evaluate mathematical statements — specifically, mathematical functions in the pre-calculus domain — instead of normal machine code.

> **Why the name "Paramath"?**
> The word "paramath" is a portmanteau of "parenthesis" and "mathematics."

### Key Features

- **Infix and prefix syntax** — arithmetic operators use infix notation; all other functions use prefix
- **Automatic optimization** — duplicate expression extraction, algebraic identity simplification
- **Loop unrolling** — compile-time iteration for performance
- **Functions** — reusable mathematical building blocks
- **Substitutions** — clean variable bindings instead of globals/locals
- **Flexible output** — store results in named variables or output directly
- **Simple CLI** — improved command-line experience over Paramath v2

---

## Syntax Overview

### Infix vs Prefix

Paramath uses both **infix** and **prefix** notation. Arithmetic operators and comparators are infix; all other functions are prefix.

Full list of infix operators: `==  !=  >  <  >=  <=  +  -  *  /  **`

```scheme
# Infix
a + b * c

# Prefix
sin(pi * x)
max(a, b)
```

### Comments

Comments start with `#` and run to end of line:

```scheme
# This is a comment
2 + 3  # This adds 2 and 3
```

### Variables

Reserved variable names (case-insensitive): single letters `a b c d e f x y z m`, plus special names `ans`, `pi`, and `euler`.

> **Note on `ans`:** The `ans` variable stores the result of the last computed expression. When `//dupe true` is enabled (default), the compiler may create intermediate variables that make `ans` behave unpredictably — prefer using named intermediates instead.

---

## Data Types & Substitutions

### Numeric Literals

```scheme
42        # Integer
3.14159   # Float
-17       # Negative integer
2.5e-3    # Scientific notation (preferred for small numbers)
```

### Substitutions

Define substitutions with the `=` token. They can reference previously defined substitutions:

```scheme
meaning = 42
golden_ratio = (1 + sqrt(5)) / 2

radius = 5
area = radius ** 2 * pi
```

### Python Expression Evaluation

Use `:=` to evaluate a Python expression at compile-time. Access previously defined substitutions via `vars.NAME`:

```scheme
length := len("hello world")

radius = 5
area := vars.radius ** 2 * pi

# In loops
repeat 5 i
    squared = vars.i ** 2
    return squared + 1
```

Available in the eval namespace: all math functions (`sin cos tan asin acos atan abs pi`), all Python built-ins (`len str int float`…), and substitutions via `vars.NAME`.

> **Warning:** Use `vars.NAME` — **not** `var.NAME`. Python-evaluated substitutions are compiled literally, so any runtime variable references will raise an error.

---

## Operators & Functions

### Basic Arithmetic

| Operator | Description    | Example        |
| -------- | -------------- | -------------- |
| `+`      | Addition       | `2 + 3` → 5    |
| `-`      | Subtraction    | `10 - 3` → 7   |
| `*`      | Multiplication | `4 * 5` → 20   |
| `/`      | Division       | `10 / 2` → 5   |
| `**`     | Exponentiation | `2 ** 8` → 256 |

> **Note:** Paramath uses standard operator precedence. Use parentheses to enforce specific evaluation order.

### Trigonometric Functions

| Function | Description     |
| -------- | --------------- |
| `sin`    | Sine            |
| `cos`    | Cosine          |
| `tan`    | Tangent         |
| `arcsin` | Inverse sine    |
| `arccos` | Inverse cosine  |
| `arctan` | Inverse tangent |
| `abs`    | Absolute value  |

```scheme
sin(pi * (x / 2))  # sin(πx/2)
```

### Logical Operations

All logical operations are continuous approximations that depend on the current `epsilon` value. They return values between 0 and 1, **not** strict booleans.

| Operation     | Syntax   | Description               |
| ------------- | -------- | ------------------------- |
| Equality      | `a == b` | ~1 when a≈b, ~0 otherwise |
| Equal to zero | `=0(a)`  | ~1 when a≈0, ~0 otherwise |
| Greater than  | `a > b`  | ~1 when a>b, ~0 otherwise |
| Greater/equal | `a >= b` | ~1 when a≥b, ~0 otherwise |
| Less than     | `a < b`  | ~1 when a<b, ~0 otherwise |
| Less/equal    | `a <= b` | ~1 when a≤b, ~0 otherwise |
| Logical not   | `!(a)`   | Returns 1−a               |

> **Tip:** Smaller epsilon = sharper comparisons. Recommended: `1e-99` for maximum sharpness.

> **Note:** `=0` and `!` are prefix functions requiring parentheses, not infix operators.

### Mathematical Operations

| Operation      | Syntax      | Description                      |
| -------------- | ----------- | -------------------------------- |
| Sign           | `sign(a)`   | Returns −1, 0, or 1 (continuous) |
| Max            | `max(a, b)` | Maximum of a and b               |
| Min            | `min(a, b)` | Minimum of a and b               |
| Max of a and 0 | `max0(a)`   | Maximum of a and 0               |
| Min of a and 0 | `min0(a)`   | Minimum of a and 0               |
| Modulo         | `mod(a, b)` | a modulo b                       |
| Fractional     | `frac(a)`   | Fractional part of a             |
| Floor          | `floor(a)`  | Floor of a                       |
| Ceiling        | `ceil(a)`   | Ceiling of a                     |
| Round          | `round(a)`  | Rounds to nearest integer        |

### Conditional Operations

`if` is a prefix function, not an operator:

```scheme
if(condition, then_value, else_value)

# Collatz conjecture step
if(mod(x, 2) == 0,
    x / 2,
    3 * x + 1
)
```

> **Warning:** Conditions are not checked for strict 0/1. Values outside [0, 1] produce blended outputs — use with caution.

---

## Program Structure

### Minimal Program

```scheme
return 2 + 2
```

### Complete Program Structure

```scheme
# Configuration
//variables a b c d e f x y z m

//precision 6
//epsilon 1e-99
//simplify true
//dupe true 0
//sympy false

# Substitutions
radius = 5

# Functions
def circle_area r
    return pi * r * r

# Output
return circle_area(radius)
```

---

## Pragmas

Pragmas start with `//` and configure compiler behaviour. They can appear anywhere and affect all subsequent code.

### `//precision`

Sets decimal precision for Python evaluation:

```scheme
//precision 4
pi_low := pi     # evaluates to 3.1416

//precision 10
pi_high := pi    # evaluates to 3.1415926536
```

> **Note:** Python's `math` module provides up to 15 significant decimals. For higher precision, define a custom `PI_ACCURATE` substitution. Using `pi` as a variable is also recommended — the compiler emits the symbol rather than the literal, keeping output shorter (e.g. `4 * pi` vs `4 * 3.141592`).

### `//epsilon`

Sets the epsilon parameter for all logical operations:

```scheme
//epsilon 1e-99   # sharp comparisons (recommended)
//epsilon 1e-10   # slightly softer
```

### `//variables`

Overrides the set of recognised variable names. Cannot change `ans`, `pi`, or `euler`:

```scheme
//variables a b c d e f x y z m   # default
//variables m n o p q             # replaces previous declaration
```

### `//simplify`

Enables or disables compile-time simplification of literals and algebraic identities:

```scheme
//simplify true    # default — simplifies 2 + 3 → 5 at compile time
//simplify false   # keep expressions as-is
```

### `//dupe`

Enables or disables duplicate subexpression extraction. Optional second parameter sets a minimum savings threshold:

```scheme
//dupe true        # default — extract repeated subexpressions
//dupe false       # keep expression as-is
//dupe true -999   # extract every possible duplication
```

> **Warning:** When `//dupe true`, the compiler may create hidden intermediates. This makes `ans` less predictable — use named variables instead.

### `//sympy`

Enables SymPy-based simplification of output expressions:

```scheme
//sympy true    # simplify via SymPy
//sympy false   # default
```

> **Warning:** Experimental. Requires SymPy (`pip install sympy`), otherwise the compiler will crash.

### Output Directives

Use `return` to emit a value, and `out <var>` before it to store the result in a named variable:

```scheme
# Emit without storing
return 2 + 3

# Store in x, then use it
out x
return 2 + 3

return x * 2
```

Default variable names: `a b c d e f x y z m`

---

## Function Definitions

### Syntax

```scheme
def function_name param1 param2 ...
    # optional substitutions
    return <expression>
```

Function names are case-sensitive. The body must end with `return`.

### Examples

```scheme
def square x
    return x * x

def distance x1 y1 x2 y2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

return distance(0, 0, 3, 4)  # → 5.0
```

> **Warning:** Recursion is strictly prohibited. Functions are expanded at compile time — a recursive call will exhaust Python's call stack.

---

## Loops & Iteration

Loops are **unrolled at compile time**. The body is copied `count` times with substitutions inlined.

### Basic Syntax

```scheme
repeat count_expression [iterator_variable]
    # body
    return <expression>
```

### Simple Loop

```scheme
size = 3

repeat size
    return x + 1
# → produces 3 separate (x + 1) expressions
```

### Loop with Iterator Variable

The iterator takes values 0, 1, … N−1:

```scheme
repeat 5 i
    return i ** 2
# → 0, 1, 4, 9, 16
```

### Local Variables in Loops

```scheme
repeat 3 i
    square = i ** 2
    return square + 1
```

### Nested Loops

```scheme
repeat 3 i
    repeat 3 j
        return i * 10 + j
```

### Advanced Example

```scheme
N = 10

repeat N i
    x_val = i + 1
    y_val = x_val ** 2
    return y_val / x_val
```

---

## Intermediates & Code Blocks

### Intermediate Assignments

Break complex expressions into named parts instead of writing deeply nested parentheses:

```scheme
# Instead of: sqrt(x ** 2 + y ** 2)
a = x ** 2
b = y ** 2
c = a + b
return c ** 0.5
```

### Naming Rules

- Names must be alphanumeric (plus underscores)
- Cannot conflict with: global constants, function names, variable names, or other intermediates in the same scope

### Complex Example

```scheme
# Quadratic formula (positive root)
discriminant = b ** 2 - 4 * a * c
sqrt_disc    = discriminant ** 0.5
numerator    = (-1 * b) + sqrt_disc
denominator  = 2 * a
return numerator / denominator
```

---

## Advanced Features

### Automatic Duplicate Detection

When `//dupe true` (default), the compiler finds repeated subexpressions and extracts them into intermediates automatically:

```scheme
//dupe true -999
return (x + 1) * (x + 1) + (x + 1) * (x + 1)
# (x + 1) appears 4 times — compiler creates an intermediate
```

### Epsilon in Expressions

Access the current epsilon value directly using `epsilon`:

```scheme
//epsilon 0.01
return x / (x + epsilon)
```

### Expression Length Optimisation

- Extracts beneficial duplicates to save space
- Flattens associative operations
- Applies algebraic identity simplification (e.g. `x * 1` → `x`)

---

## Complete Examples

### Polynomial Evaluation (Horner's Method)

```scheme
//precision 6
//epsilon 1e-99

def horner x a b c d
    t1 = a * x
    t2 = t1 + b
    t3 = t2 * x
    t4 = t3 + c
    t5 = t4 * x
    return t5 + d

return horner(x, 2, -3, 1, 5)
```

### Vector Dot Product & Angle

```scheme
def dot_product ax ay bx by
    return ax * bx + ay * by

def magnitude vx vy
    return (vx * vx + vy * vy) ** 0.5

dot  = dot_product(a, b, c, d)
mag1 = magnitude(a, b)
mag2 = magnitude(c, d)
return dot / (mag1 * mag2)  # cosine of angle
```

### Smooth Activation Functions

```scheme
//epsilon 1e-99

def sigmoid x
    return 1 / (1 + euler ** (-1 * x))

def relu x
    return max0(x)

def gelu x
    sig = sigmoid(1.702 * x)
    return x * sig

return gelu(x)
```

### Parametric Circle

```scheme
//precision 8
//epsilon 1e-99

N = 50

repeat N i
    t    = i / N
    xpos = cos(2 * pi * t)
    ypos = sin(2 * pi * t)
    return xpos * xpos + ypos * ypos  # should be 1 everywhere
```

---

## Command-Line Interface

### Basic Usage

```bash
paramath [options] filepath
```

### Options

| Flag      | Long Form        | Description                             |
| --------- | ---------------- | --------------------------------------- |
| `-h`      | `--help`         | Show help message                       |
| `-v`      | `--version`      | Print version and exit                  |
| `-o FILE` | `--output FILE`  | Output file (default: `math.txt`)       |
| `-D`      | `--debug`        | Enable debug output                     |
| `-V`      | `--verbose`      | Enable verbose output                   |
| `-O`      | `--print-output` | Print compiled output to console        |
| `-m`      | `--math-output`  | Format output for calculators           |
| `-S`      | `--safe-eval`    | Block Python eval; print what would run |
| `-L FILE` | `--logfile FILE` | Write logs to FILE                      |

### Flag Details

#### `-O` / `--print-output`

Prints compiled expressions to stdout in addition to writing to the output file.

#### `-S` / `--safe-eval`

Security feature: blocks execution of Python `:=` expressions. Use this when inspecting untrusted `.pm` files before running them.

#### `-D` / `--debug`

Enables detailed debug output showing parsing steps, AST transformations, and optimisation decisions.

#### `-V` / `--verbose`

Enables high-level compilation progress output.

### Examples

```bash
# Compile with verbose output, custom output file
paramath myprogram.pm -V -o results.txt

# Print compiled output and save a debug log
paramath myprogram.pm -O -D -L debug.log

# Inspect an untrusted file without executing Python
paramath untrusted.pm -S

# Minimal quiet compile
paramath myprogram.pm
```

---

## Error Reference

### Parser Errors

| Error                            | Cause                                       | Fix                           |
| -------------------------------- | ------------------------------------------- | ----------------------------- |
| `Unexpected end of tokens`       | Missing closing `)`                         | Add `)`                       |
| `Unexpected closing parenthesis` | Extra `)` or missing `(`                    | Check balance                 |
| `Undefined identifier 'X'`       | Unknown variable                            | Define or check spelling      |
| `Malformed expression: <expr>`   | Malformed expression or wrong function call | Check syntax                  |
| `Empty expression`               | `return` with no body                       | Add expression after `return` |
| `Unknown statement: X`           | Unrecognised top-level token                | Check statement syntax        |

### Function Errors

| Error                                     | Cause                        | Fix                       |
| ----------------------------------------- | ---------------------------- | ------------------------- |
| `Function 'X' expects N args, got M`      | Wrong argument count         | Check function definition |
| `Function 'X' missing return statement`   | No `return` in function body | Add `return`              |
| `Recursive substitution detected for 'X'` | Circular substitution        | Break the cycle           |

### Loop Errors

| Error                                           | Cause                       | Fix                               |
| ----------------------------------------------- | --------------------------- | --------------------------------- |
| `Repetition count must be an integer`           | Float or non-constant count | Use a constant integer            |
| `Precision must be set before repeat statement` | Missing `//precision`       | Add `//precision` before `repeat` |
| `Epsilon must be set before repeat statement`   | Missing `//epsilon`         | Add `//epsilon` before `repeat`   |
| `Iteration variable 'X' conflicts with …`       | Name collision              | Rename the variable               |

### Pragma Errors

| Error                                               | Cause                                | Fix                        |
| --------------------------------------------------- | ------------------------------------ | -------------------------- |
| `Unknown pragma: X`                                 | Typo or unsupported pragma           | Check pragma name          |
| `Precision must be an integer`                      | Non-integer value                    | Use e.g. `//precision 6`   |
| `Epsilon must be a float`                           | Non-numeric value                    | Use e.g. `//epsilon 1e-99` |
| `Dupe min_savings must be an integer`               | Non-integer second arg               | Use e.g. `//dupe true 0`   |
| `'X' conflicts with a name declared in //variables` | Substitution name matches a variable | Rename the substitution    |

---

## Best Practices

### 1. Use Small Epsilon Values

```scheme
//epsilon 1e-99  # recommended for sharp comparisons
```

> **Tip:** Avoid epsilon values larger than `1e-50` unless you specifically need smooth, gradient-friendly transitions.

### 2. Prefer Intermediates Over Nested Expressions

```scheme
# Hard to read
return ((x2-x1)**2 + (y2-y1)**2) / (abs(x2-x1) + abs(y2-y1))

# Clear and maintainable
dx        = x2 - x1
dy        = y2 - y1
euclidean = (dx * dx + dy * dy) ** 0.5
manhattan = abs(dx) + abs(dy)
return euclidean / manhattan
```

### 3. Avoid Relying on `ans` with Dupe Enabled

When `//dupe true`, hidden intermediates are created that overwrite `ans`. Use `out <var>` to store results explicitly:

```scheme
out a
return some_complex_expression(x)

return a * 2  # safe — a is explicit
```

### 4. Use Loops for Repeated Patterns

```scheme
# Bad: copy-paste
return 0 ** 2
return 1 ** 2
return 2 ** 2

# Good: loop
repeat 100 i
    return i ** 2
```

### 5. Break Complex Functions into Smaller Ones

```scheme
def sigmoid x
    return 1 / (1 + euler ** (-1 * x))

def layer x w b
    return sigmoid(w * x + b)

def network x
    h1 = layer(x,   0.5,  0.1)
    h2 = layer(h1,  0.3, -0.2)
    return layer(h2, 0.8,  0.0)
```

### 6. Comment Your Epsilon Choices

```scheme
# Sharp comparison for exact equality check
//epsilon 1e-99
return x == y

# Smooth transition for gradient-based optimisation
//epsilon 1e-10
return if(x > threshold, high_val, low_val)
```

---

_Paramath v3.0.12 · MIT License · [github.com/kaemori/paramath3](https://github.com/kaemori/paramath3)_
