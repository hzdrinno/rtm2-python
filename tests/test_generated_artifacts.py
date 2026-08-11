import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtm2 import RTM2  # noqa: E402


class GeneratedArtifactTests(unittest.TestCase):
    def test_type_stub_is_current(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            generated = Path(temp_dir) / "__init__.pyi"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "generate_rtm2_pyi.py"),
                    "--output",
                    str(generated),
                    "--no-py-typed",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            committed = (ROOT / "src" / "rtm2" / "__init__.pyi").read_text(encoding="utf-8")
            self.assertEqual(generated.read_text(encoding="utf-8"), committed)

    def test_rawd_type_stub_accepts_one_or_two_integers(self):
        stub = (ROOT / "src" / "rtm2" / "__init__.pyi").read_text(encoding="utf-8")
        self.assertIn("def rawd(self, arg0: int) -> None: ...", stub)
        self.assertIn("def rawd(self, arg0: int, arg1: int) -> None: ...", stub)

    def test_command_reference_covers_registry(self):
        reference = (ROOT / "docs" / "commands.md").read_text(encoding="utf-8")

        for command in RTM2._COMMANDS:
            with self.subTest(command=command):
                self.assertIn(f"| `{command}` |", reference)


if __name__ == "__main__":
    unittest.main()
