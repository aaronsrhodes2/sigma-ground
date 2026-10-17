"""scripts/jobwatch.py: the watchdog detects the ways an expensive job goes wrong, and stays quiet when it does not.

Each test launches a tiny stand-in job as a real subprocess and a real watchdog with a 1 s interval, then checks the watchdog's exit code and message:
clean finish (0, DONE), process killed without a success marker (2), a stalled job that stays alive but stops progressing (2), a traceback in the log (2),
retry noise (a warning, no exit), and a healthy job that keeps progressing and is NOT flagged. Ground truth is the behaviour scripted into the stand-in job,
never the watchdog's own output.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

JOBWATCH = Path(__file__).resolve().parent.parent / "scripts" / "jobwatch.py"
NL = "\\n"                                    # a backslash-n written INTO the stand-in script's source, so it prints a newline when the job runs


def start_job(tmp: Path, lines: list[str]) -> subprocess.Popen:
    (tmp / "job.py").write_text("\n".join(lines) + "\n")
    return subprocess.Popen([sys.executable, str(tmp / "job.py")], cwd=tmp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def log_line(text: str) -> str:
    return f"open('job.log','a').write('{text}{NL}')"


def watch_cmd(tmp: Path, pid: int, *extra: str) -> list[str]:
    return [sys.executable, str(JOBWATCH), "--name", "t", "--pid", str(pid), "--interval", "1", "--state-dir", str(tmp / "state"),
            "--log", str(tmp / "job.log"), "--progress", f"lines:{tmp / 'job.log'}", "--min-children", "0", *extra]


class TestJobWatch(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        (self.tmp / "job.log").write_text("")

    def tearDown(self):
        self._td.cleanup()

    def test_clean_finish_is_done_not_alert(self):
        job = start_job(self.tmp, ["import time", "for i in range(4):", "    " + log_line("step done"), "    time.sleep(1)",
                                   log_line("finished: 4 ok, 0 failed")])
        r = subprocess.run(watch_cmd(self.tmp, job.pid, "--from-start"), capture_output=True, text=True, timeout=40)
        job.wait()
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("JOBWATCH DONE", r.stdout)

    def test_a_killed_job_without_a_success_marker_is_an_alert(self):
        job = start_job(self.tmp, ["import time", log_line("step 0 done"), "time.sleep(60)"])
        time.sleep(1.5)
        w = subprocess.Popen(watch_cmd(self.tmp, job.pid), stdout=subprocess.PIPE, text=True)
        time.sleep(2.5)
        job.kill()
        out, _ = w.communicate(timeout=30)
        self.assertEqual(w.returncode, 2, out)
        self.assertIn("driver process is gone", out)

    def test_a_stalled_job_is_an_alert(self):
        job = start_job(self.tmp, ["import time", log_line("step 0 done"), "time.sleep(60)"])
        try:
            r = subprocess.run(watch_cmd(self.tmp, job.pid, "--stall-minutes", "0.05"), capture_output=True, text=True, timeout=40)   # 3 s without a change
        finally:
            job.kill()
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("no progress signal changed", r.stdout)

    def test_a_traceback_in_the_log_is_an_alert(self):
        job = start_job(self.tmp, ["import time", "time.sleep(2)", log_line("Traceback (most recent call last):"), "time.sleep(60)"])
        try:
            r = subprocess.run(watch_cmd(self.tmp, job.pid), capture_output=True, text=True, timeout=40)
        finally:
            job.kill()
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("Traceback", r.stdout)

    def test_a_healthy_progressing_job_is_not_flagged(self):                  # the watchdog must be able to stay quiet
        job = start_job(self.tmp, ["import time", "for i in range(6):", "    " + log_line("step done"), "    time.sleep(1)", "time.sleep(60)"])
        try:
            r = subprocess.run(watch_cmd(self.tmp, job.pid, "--stall-minutes", "0.1", "--max-hours", "0.0008"), capture_output=True, text=True, timeout=40)
        finally:
            job.kill()
        self.assertEqual(r.returncode, 3, r.stdout)                            # 3 = watched long enough, still healthy (not an error)
        self.assertIn("TIMEOUT", r.stdout)

    def test_retry_noise_is_a_warning_not_an_alert(self):
        job = start_job(self.tmp, ["import time", "time.sleep(1.5)", log_line("FAILED job 3 exit code 1, attempt 1/3; retrying"), "time.sleep(60)"])
        try:
            r = subprocess.run(watch_cmd(self.tmp, job.pid, "--stall-minutes", "5", "--max-hours", "0.001"), capture_output=True, text=True, timeout=40)
        finally:
            job.kill()
        self.assertEqual(r.returncode, 3, r.stdout)
        self.assertIn('"kind": "WARN"', (self.tmp / "state" / "t.jsonl").read_text())

    def test_an_idle_stalled_job_is_reported_as_idle(self):
        job = start_job(self.tmp, ["import time", log_line("step 0 done"), "time.sleep(60)"])
        try:
            r = subprocess.run(watch_cmd(self.tmp, job.pid, "--stall-minutes", "0.05"), capture_output=True, text=True, timeout=40)
        finally:
            job.kill()
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("IDLE", r.stdout)

    def test_a_busy_job_without_progress_gets_a_grace_period_then_an_alert(self):
        # a stand-in that burns CPU but writes nothing: a long computation or a loop. The limit (3 s) is extended once, then it alerts as BUSY.
        job = start_job(self.tmp, [log_line("step 0 done"), "while True:", "    pass"])
        t0 = time.time()
        try:
            r = subprocess.run(watch_cmd(self.tmp, job.pid, "--stall-minutes", "0.05", "--busy-grace", "1"), capture_output=True, text=True, timeout=60)
        finally:
            job.kill()
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("BUSY", r.stdout)
        self.assertGreaterEqual(time.time() - t0, 5.5)                       # alerted after the extended (2 x 3 s) limit, not at the first
        self.assertIn("limit extended", (self.tmp / "state" / "t.jsonl").read_text())

    def test_expected_job_minutes_raises_the_stall_limit(self):
        job = start_job(self.tmp, ["import time", log_line("step 0 done"), "time.sleep(60)"])
        t0 = time.time()
        try:
            r = subprocess.run(watch_cmd(self.tmp, job.pid, "--stall-minutes", "0.01", "--expected-job-minutes", "0.05"), capture_output=True, text=True, timeout=60)
        finally:
            job.kill()
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertGreaterEqual(time.time() - t0, 6.5)                       # limit = 2.5 x 0.05 min = 7.5 s, not 0.6 s

    def test_fair_preset_recognises_the_validate_mode_success_file(self):
        # regression: a --validate run writes fair_contest_validate.json; the preset once looked only for fair_contest_contest.json and cried wolf at a clean finish
        out = self.tmp / "run"
        (out / "jobs").mkdir(parents=True)
        job = start_job(self.tmp, ["import time", "time.sleep(2)"])
        (out / "fair_contest_validate.json").write_text("{}")
        r = subprocess.run([sys.executable, str(JOBWATCH), "--name", "t", "--pid", str(job.pid), "--interval", "1", "--state-dir", str(self.tmp / "state"),
                            "--preset", "fair", "--outdir", str(out), "--min-children", "0", "--stall-minutes", "5"], capture_output=True, text=True, timeout=40)
        job.wait()
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("JOBWATCH DONE", r.stdout)

    def test_a_single_process_job_needs_no_flag_to_avoid_a_children_alarm(self):      # BUG 2026-10-16: the default min-children of 1 raised a false alarm on a trickle job (one process, no workers)
        job = start_job(self.tmp, ["import time", "for i in range(6):", "    " + log_line("step done"), "    time.sleep(1)", log_line("all finished")])
        cmd = [sys.executable, str(JOBWATCH), "--name", "t", "--pid", str(job.pid), "--interval", "1", "--state-dir", str(self.tmp / "state"), "--log", str(self.tmp / "job.log"),
               "--progress", f"lines:{self.tmp / 'job.log'}", "--success-regex", "all finished", "--from-start"]          # no --min-children and no preset
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        job.wait()
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertNotIn("python children", r.stdout)

    def test_a_pid_that_is_not_running_is_an_alert_at_once(self):
        r = subprocess.run(watch_cmd(self.tmp, 4000000), capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 2)
        self.assertIn("none of the PIDs", r.stdout)


if __name__ == "__main__":
    unittest.main()
