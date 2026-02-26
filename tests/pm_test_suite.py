from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paramath3


PM3_TEST_DIR = ROOT / "paramath_test"


def _iter_pm3_files() -> list[Path]:
    return sorted(PM3_TEST_DIR.rglob("*.pm3"))


def _test_name_from_relpath(rel_path: Path) -> str:
    return "_".join(rel_path.with_suffix("").parts)


class Pm3FixtureSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paramath3.VERBOSE = False
        paramath3.DEBUG = False
        paramath3.LOGFILE = None

    def _compile_source(self, source: str):
        code = paramath3.code_to_lines(source)
        asts = paramath3.parse_pm3_to_ast(code, progress=False)
        return paramath3.process_asts(asts, progress=False)

    def _compile_pm3_file(self, pm3_path: Path):
        source = pm3_path.read_text(encoding="utf-8")
        return self._compile_source(source)

    def test_pm3_fixture_directory_exists(self):
        self.assertTrue(
            PM3_TEST_DIR.exists(), msg=f"Missing pm3 fixture directory: {PM3_TEST_DIR}"
        )

    def test_pm3_fixture_directory_contains_files(self):
        self.assertGreater(
            len(_iter_pm3_files()),
            0,
            msg="pm3 fixture directory contains no .pm3 files",
        )


def _make_case_test(pm3_path: Path):
    rel = pm3_path.relative_to(PM3_TEST_DIR)
    case_name = str(rel)

    def _test(self: Pm3FixtureSuiteTests):
        if rel.name == "sympy_integration.pm3":
            try:
                import sympy  # noqa: F401
            except Exception:
                self.skipTest("sympy is not installed in this environment")

        outputs = self._compile_pm3_file(pm3_path)
        self.assertGreater(
            len(outputs), 0, msg=f"No outputs generated in case: {case_name}"
        )

        for output_var, expression in outputs:
            self.assertTrue(
                output_var is None or isinstance(output_var, str),
                msg=f"Output variable has unexpected type in case {case_name}",
            )
            self.assertIsInstance(
                expression, str, msg=f"Expression is not string in case {case_name}"
            )
            self.assertNotEqual(
                expression.strip(), "", msg=f"Empty expression in case {case_name}"
            )

    return _test


for _pm3 in _iter_pm3_files():
    _rel = _pm3.relative_to(PM3_TEST_DIR)
    setattr(
        Pm3FixtureSuiteTests,
        f"test_pm3_fixture_{_test_name_from_relpath(_rel)}",
        _make_case_test(_pm3),
    )


if __name__ == "__main__":
    unittest.main()
