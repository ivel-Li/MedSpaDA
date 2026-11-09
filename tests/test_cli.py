import os
from pathlib import Path
import subprocess
import sys
import unittest


class CLITests(unittest.TestCase):
    def test_public_entry_points_have_help_without_weights_or_data(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy(); env.pop("MEDSPADA_MANIFEST", None)
        env["PYTHONPATH"] = str(root)
        for script in ("train", "evaluate", "infer", "validate_data", "make_synthetic_data"):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, str(root/"scripts"/f"{script}.py"), "--help"],
                                        cwd=root, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
