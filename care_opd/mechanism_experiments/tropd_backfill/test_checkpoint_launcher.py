"""CPU-only launcher/protocol interface checks, no real GPU inference."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from test_gpu67_scheduler import BASH, _shell_path
import protocol_guard

HERE = Path(__file__).resolve().parent


@unittest.skipUnless(BASH, "bash unavailable")
class CheckpointLauncherTest(unittest.TestCase):
    def plan(self, launcher, **env):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "new-output"
            process = subprocess.run(
                [BASH, _shell_path(HERE / launcher), "plan" if launcher.startswith("run_qzbc_gpu1") else "qzbc"],
                env={**os.environ, "DRY_RUN": "1", "GPU1_CHECKPOINT_ROOT": _shell_path(root),
                     "BACKFILL_ROOT": _shell_path(root), **env},
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertFalse(root.exists(), "plan should not create run roots")
            return process.stdout

    def test_gpu1_plan_is_qzbc_only_and_ignores_old_lane_environment(self):
        output = self.plan("run_qzbc_gpu1_checkpoint.sh", PHYSICAL_GPU="6", QZBC_CHECKPOINT="0",
                           QZBC_OUT_ROOT="/do-not-use-old-output")
        self.assertEqual(output.count("QZBC_CMD "), 2)
        self.assertEqual(output.count("--checkpoint-contract"), 2)
        self.assertIn("physical_gpu=1", output)
        self.assertNotIn("REAL_CMD", output)
        self.assertNotIn("do-not-use-old-output", output)
        self.assertIn("--max-new-tokens 1200 --greedy --seed 42", output)

    def test_legacy_default_still_has_no_checkpoint_flag(self):
        output = self.plan("run_tropd_backfill.sh", QZBC_CHECKPOINT="0")
        self.assertNotIn("--checkpoint-contract", output)

    def test_checkpoint_optin_prints_exact_contract_argument(self):
        output = self.plan("run_tropd_backfill.sh", QZBC_CHECKPOINT="1")
        self.assertEqual(output.count("--checkpoint-contract"), 2)
        self.assertEqual(output.count("protocol_lock.json"), 2)

    def test_scripts_and_documented_shell_compile(self):
        for name in ("run_tropd_backfill.sh", "run_qzbc_gpu1_checkpoint.sh"):
            subprocess.run([BASH, "-n", _shell_path(HERE / name)], check=True)
        document = (HERE / "SERVER_COMMANDS_GPU1_CHECKPOINT.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```bash\n(.*?)```", document, re.S)
        self.assertEqual(len(blocks), 4)
        for block in blocks:
            subprocess.run([BASH, "-n"], input=block, text=True, check=True)


class CheckpointProtocolInterfaceTest(unittest.TestCase):
    def test_optional_guard_flag(self):
        args = ["prepare-qzbc"]
        for name in ("project-root", "qzbc-root", "requests", "seed-manifest", "compiled-train-manifest",
                     "compiled-dev-manifest", "compiled-test-manifest", "model-lock", "output-dir"):
            args.extend(["--" + name, "/synthetic"])
        args.extend(["--physical-gpu", "1", "--logical-device", "cuda:0"])
        parser = protocol_guard.make_parser()
        self.assertFalse(parser.parse_args(args).durable_checkpoint)
        self.assertTrue(parser.parse_args([*args, "--durable-checkpoint"]).durable_checkpoint)


if __name__ == "__main__":
    unittest.main()
