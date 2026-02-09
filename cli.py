#!/usr/bin/env python3
import time

time_start = time.time()
import os

import_os_time = time.time() - time_start
last_time = time.time()
import builtins

import_builtins_time = time.time() - last_time
last_time = time.time()
import argparse

import_argparse_time = time.time() - last_time
last_time = time.time()
import sys

import_sys_time = time.time() - last_time
last_time = time.time()
from colors import colors

import_colors_time = time.time() - last_time
last_time = time.time()
from pm3 import (
    parse_pm3_to_ast,
    process_asts,
    check_python_eval,
    print_debug,
    print_verbose,
)
import pm3.logging as pm3_logging

import_pm3_time = time.time() - last_time

PROGRAM_VERSION = "3.0.12"


def _supports_color(stream) -> bool:
    if os.getenv("NO_COLOR") is not None:
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def _cli_color_enabled() -> bool:
    # allow user to force-disable without waiting for argparse
    if "--no-color" in sys.argv:
        return False
    return _supports_color(sys.stdout)


COLOR_ENABLED = _cli_color_enabled()


def _c(code: str, text: str) -> str:
    if not COLOR_ENABLED:
        return text
    return f"{code}{text}{colors.Reset.reset}"


def _title(text: str) -> str:
    return _c(colors.BoldHighIntensity.cyan, text)


def _ok(text: str) -> str:
    return _c(colors.BoldHighIntensity.green, text)


def _warn(text: str) -> str:
    return _c(colors.BoldHighIntensity.yellow, text)


def _err(text: str) -> str:
    return _c(colors.BoldHighIntensity.red, text)


def _dim(text: str) -> str:
    return _c(colors.HighIntensity.black, text)


def _path(text: str) -> str:
    return _c(colors.HighIntensity.cyan, text)


def _clear(text: str) -> str:
    return _c(colors.Reset.reset, text)


def main():
    parser = argparse.ArgumentParser(
        description="Paramath Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  paramath3 testfile.pm
  paramath3 testfile.pm -dv
  paramath3 testfile.pm -sL output.log
  paramath3 testfile.pm -dvL debug.log
  paramath3 testfile.pm -tmo compiled.txt -O
  paramath3 testfile.pm -mtpo compiled.txt
        """,
    )

    parser.add_argument(
        "filepath",
        nargs="?",
        help="Input paramath file",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"Paramath {PROGRAM_VERSION}",
        help="prints the Paramath version number and exits",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="math.txt",
        metavar="FILE",
        help="output file (default: math.txt)",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable debug output"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose output"
    )
    parser.add_argument(
        "-s", "--silent", action="store_true", help="suppress all output"
    )
    parser.add_argument(
        "-O",
        "--print-output",
        action="store_true",
        help="print out the compiled output",
    )
    parser.add_argument(
        "-m",
        "--math-output",
        action="store_true",
        help="format output for calculators (use ^, implicit multiplication, capital ANS)",
    )
    parser.add_argument(
        "-t",
        "--trust",
        action="store_true",
        help="enable unsafe code execution (default: disabled) (WARNING: may be dangerous! use at your own risk!)",
    )
    parser.add_argument(
        "-L",
        "--logfile",
        required=False,
        metavar="FILE",
        help="write logs from verbose or debug to FILE",
    )
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="show parsing progress using tqdm (WARNING: may slow down parsing)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable colored CLI output",
    )

    # nts: verbose means human readable logs
    # debug means less-human-readable, debugging-focused logs
    args = parser.parse_args()
    verbose = args.verbose
    debug = args.debug
    logfile = args.logfile
    trust = args.trust
    progress = args.progress
    math_output = args.math_output
    output_file = args.output
    print_output = args.print_output
    input_file = args.filepath

    old_print = None
    if args.silent:
        # just monkeypatch print to do nothing
        old_print = builtins.print
        builtins.print = lambda *a, **k: None

    print(_title(f"Paramath v{PROGRAM_VERSION}") + "\n")

    global COLOR_ENABLED
    if args.no_color:
        COLOR_ENABLED = False

    if args.filepath is None:
        parser.print_help()
        return

    if progress and (debug or verbose):
        raise ValueError("Progress bar cannot be used with verbose or debug modes")

    if verbose:
        pm3_logging.VERBOSE = True
        print_verbose("verbose logging enabled")

    if debug:
        pm3_logging.DEBUG = True
        print_debug("debug logging enabled")

    if logfile:
        pm3_logging.LOGFILE = logfile
        print_verbose(f'Logging to file: "{logfile}"')
        with open(logfile, "w") as f:
            f.write(
                f"Paramath v{PROGRAM_VERSION} Logfile (at {time.strftime('%Y-%m-%d %H:%M:%S')})\n\n"
            )
        print_debug(f'Logfile "{logfile}" initialized.')

    print_debug(
        "argv/options: "
        + str(
            {
                "input": input_file,
                "output": output_file,
                "progress": progress,
                "trust": trust,
                "math_output": math_output,
                "print_output": print_output,
                "verbose": verbose,
                "debug": debug,
                "logfile": logfile,
            }
        )
    )

    print_debug(f"Import os time: {import_os_time*1000:.6f}ms")
    print_debug(f"Import builtins time: {import_builtins_time*1000:.6f}ms")
    print_debug(f"Import argparse time: {import_argparse_time*1000:.6f}ms")
    print_debug(f"Import sys time: {import_sys_time*1000:.6f}ms")
    print_debug(f"Import colors time: {import_colors_time*1000:.6f}ms")
    print_debug(f"Import pm3 time: {import_pm3_time*1000:.6f}ms")
    print_debug(f"Init finished: {(time.time() - time_start)*1000:.6f}ms")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            code = [line.rstrip() for line in f.readlines() if line]

            print_verbose(
                f"loaded script: {len(code)} non-empty lines from '{input_file}'"
            )

            if not trust:
                print(
                    _warn("Running as untrusted.")
                    + " "
                    + _dim("No code will be executed.")
                )
                python_eval = check_python_eval(code)
                if python_eval:
                    if old_print is not None:
                        builtins.print = old_print

                    print(_err("The following Python code was found in the script:"))
                    for line_no, line in python_eval:
                        print(_err(f"Line {line_no}:") + f' "{line}"')
                    print(
                        _err("Aborting due to untrusted code execution.")
                        + " "
                        + _dim("Review the code, THEN run with -t to trust.")
                    )
                    sys.exit(1)

                print("")

        asts, config = parse_pm3_to_ast(code, progress=progress)
        print_verbose(f"parsed to ASTs: {len(asts)} outputs")
        outputs = process_asts(asts, progress=progress)
        print_verbose(f"compiled outputs: {len(outputs)} expressions")

        with open(output_file, "w", encoding="utf-8") as f:
            for output, expr in outputs:
                if math_output:
                    output = (
                        output.replace("**", "^")
                        .replace("*", "")
                        .replace("ans", "ANS")
                        .replace("pi", "π")
                        .replace("e", "ℯ")
                    )

                f.write(f"to {output}:\n")
                f.write(expr + "\n")
                if print_output:
                    print(f"to {output}:")
                    print(expr)

        print(_ok("=== Compilation successful! ==="))
        print(_dim("Compiled in") + _ok(f" {(time.time() - time_start):.6f}s!"))
        print(_dim("Generated ") + _clear(f"{len(outputs)} outputs"))
        print(_dim("Output written to ") + _path(f'"{output_file}"'))

    except FileNotFoundError:
        if old_print is not None:
            builtins.print = old_print
        print(_err("=== Compilation failed! ==="))
        print(
            _err("Error:")
            + " "
            + _dim("Input file ")
            + _path(f'"{input_file}"')
            + _dim(" not found!")
        )

    except Exception as e:
        if old_print is not None:
            builtins.print = old_print
        print(_err("=== Compilation failed! ==="))
        print(_err("Error:") + " " + str(e))


if __name__ == "__main__":
    main()
