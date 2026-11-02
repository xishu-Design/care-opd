from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


HERE = Path(__file__).resolve().parent
SCHEDULER = HERE / "run_tropd_backfill_gpu67.sh"


def _bash() -> str | None:
    if os.name == "nt":
        for candidate in (
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
        ):
            if candidate.is_file():
                return str(candidate)
        return None
    return shutil.which("bash")


BASH = _bash()


def _shell_path(path: Path | str) -> str:
    resolved = str(Path(path).resolve())
    if os.name != "nt":
        return resolved
    assert BASH is not None
    converted = subprocess.run(
        [BASH, "-lc", 'cygpath -u "$1"', "gpu67-test", resolved],
        check=True,
        text=True,
        capture_output=True,
    )
    return converted.stdout.strip()


@unittest.skipUnless(BASH, "Git Bash/bash is unavailable")
class Gpu67SchedulerTest(unittest.TestCase):
    def _fixtures(self, root: Path) -> tuple[str, str, str, str]:
        mock = root / "mock_single_gpu_runner.sh"
        events = root / "events.log"
        fake_bin = root / "fake-bin"
        fake_bin.mkdir()
        fake_flock = fake_bin / "flock"
        fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        mock.write_text(
            textwrap.dedent(
                r"""
                #!/usr/bin/env bash
                set -u
                stage="$1"
                mkdir -p "$BACKFILL_ROOT"
                printf '%s\n' '{"contract_fingerprint":"e63c9a4f385c6dc88a491f75564a0835fc2c8518f6de0ff09168e176b06e6ae2","models":{"tropd_full":{"model_fingerprint":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"care_opd":{"model_fingerprint":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}}}' > "$BACKFILL_ROOT/MODEL_LOCK.json"
                printf '%s|%s|%s|%s|%s|%s|start\n' \
                  "$stage" "$PHYSICAL_GPU" "$BACKFILL_ROOT" \
                  "${QZBC_OUT_ROOT-__UNSET__}" "${STRUCTURED_OUT-__UNSET__}" \
                  "${RAW_OUT-__UNSET__}" >> "$MOCK_EVENT_LOG"
                parent="$(dirname "$BACKFILL_ROOT")"
                case "$stage" in
                  qzbc)
                    : > "$parent/gpu6-started"
                    sleep 0.35
                    ;;
                  structured)
                    for _ in $(seq 1 100); do
                      [[ -e "$parent/gpu6-started" ]] && break
                      sleep 0.01
                    done
                    if [[ "${FAIL_STRUCTURED:-0}" == 1 ]]; then
                      printf '%s|%s|failed\n' "$stage" "$PHYSICAL_GPU" >> "$MOCK_EVENT_LOG"
                      exit 7
                    fi
                    sleep 0.04
                    ;;
                  raw)
                    sleep 0.04
                    ;;
                esac
                printf '%s|%s|end\n' "$stage" "$PHYSICAL_GPU" >> "$MOCK_EVENT_LOG"
                """
            ).lstrip(),
            encoding="utf-8",
        )
        os.chmod(fake_flock, 0o755)
        os.chmod(mock, 0o755)
        return (
            _shell_path(mock),
            _shell_path(events),
            _shell_path(fake_bin),
            _shell_path(root / "parallel"),
        )

    def _run(self, root: Path, *, dry_run: bool, fail_structured: bool = False):
        mock, events, fake_bin, parallel = self._fixtures(root)
        scheduler = _shell_path(SCHEDULER)
        python = _shell_path(sys.executable)
        command = r'''
            PATH="$1:$PATH"
            export SINGLE_GPU_RUNNER="$2"
            export PARALLEL_ROOT="$3"
            export MOCK_EVENT_LOG="$4"
            export PY="$5"
            export DRY_RUN="$6"
            export FAIL_STRUCTURED="$7"
            export QZBC_OUT_ROOT=/stale/qzbc
            export STRUCTURED_OUT=/stale/structured
            export RAW_OUT=/stale/raw
            export MODEL_LOCK=/stale/model-lock.json
            bash "$8" all
        '''
        completed = subprocess.run(
            [
                BASH,
                "-c",
                command,
                "gpu67-test",
                fake_bin,
                mock,
                parallel,
                events,
                python,
                "1" if dry_run else "0",
                "1" if fail_structured else "0",
                scheduler,
            ],
            text=True,
            capture_output=True,
            timeout=20,
        )
        rows = Path(root / "events.log").read_text(encoding="utf-8").splitlines()
        return completed, rows, Path(root / "parallel")

    def test_bash_syntax_and_dry_run_separates_gpu_paths_and_unsets_stale_outputs(self):
        syntax = subprocess.run(
            [BASH, "-n", _shell_path(SCHEDULER)],
            text=True,
            capture_output=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        with tempfile.TemporaryDirectory() as folder:
            completed, rows, _ = self._run(Path(folder), dry_run=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        starts = [row.split("|") for row in rows if row.endswith("|start")]
        self.assertEqual([row[0] for row in starts], ["qzbc", "structured", "raw"])
        self.assertEqual([row[1] for row in starts], ["6", "7", "7"])
        self.assertTrue(starts[0][2].endswith("/gpu6_qzbc"), starts[0])
        self.assertTrue(starts[1][2].endswith("/gpu7_real"), starts[1])
        for row in starts:
            self.assertEqual(row[3:6], ["__UNSET__"] * 3)

    def test_mocked_lanes_overlap_and_gpu7_raw_follows_structured(self):
        with tempfile.TemporaryDirectory() as folder:
            completed, rows, _ = self._run(Path(folder), dry_run=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        q_start = rows.index("qzbc|6|" + next(
            row.split("|")[2] for row in rows if row.startswith("qzbc|6|")
        ) + "|__UNSET__|__UNSET__|__UNSET__|start")
        structured_start = next(i for i, row in enumerate(rows) if row.startswith("structured|7|") and row.endswith("|start"))
        structured_end = rows.index("structured|7|end")
        raw_start = next(i for i, row in enumerate(rows) if row.startswith("raw|7|") and row.endswith("|start"))
        raw_end = rows.index("raw|7|end")
        q_end = rows.index("qzbc|6|end")
        self.assertLess(q_start, structured_start)
        self.assertLess(structured_start, structured_end)
        self.assertLess(structured_end, raw_start)
        self.assertLess(raw_start, raw_end)
        self.assertLess(raw_end, q_end, "GPU6 and GPU7 lanes did not overlap")
        self.assertIn("GPU67_RESULT gpu6_rc=0 gpu7_rc=0", completed.stdout)

    def test_failed_structured_suppresses_raw_but_reaps_gpu6(self):
        with tempfile.TemporaryDirectory() as folder:
            completed, rows, parallel = self._run(
                Path(folder), dry_run=False, fail_structured=True
            )
            gpu7_log = (parallel / "gpu7_real.log").read_text(encoding="utf-8")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("qzbc|6|end", rows)
        self.assertIn("structured|7|failed", rows)
        self.assertFalse(any(row.startswith("raw|") for row in rows))
        self.assertIn("GPU67_RESULT gpu6_rc=0 gpu7_rc=7", completed.stdout)
        self.assertIn("raw not started", gpu7_log)

    def test_source_uses_coordinator_lock_and_has_no_process_kill(self):
        source = SCHEDULER.read_text(encoding="utf-8")
        self.assertIn("flock -n 8", source)
        self.assertIn('wait "$pid6" || rc6=$?', source)
        self.assertIn('wait "$pid7" || rc7=$?', source)
        for forbidden in ("pkill", "kill -9", "nvidia-smi --gpu-reset"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
