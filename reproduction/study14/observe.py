"""Own one compression wrapper and retain its signed wait and host clock interval."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

output, *argv = sys.argv[1:]
output = Path(output)
fd = int(os.environ["STUDY14_LOCK_FD"])
if not os.path.samestat(os.fstat(fd), Path("build/research-gpu0.lock").stat()):
    raise SystemExit("Canonical lock inode differs")
info = time.get_clock_info("perf_counter")
if info.implementation != "clock_gettime(CLOCK_MONOTONIC)" or info.adjustable or not info.monotonic:
    raise SystemExit("Unexpected host performance clock")
boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
with output.with_suffix(".stdout").open("xb") as stdout, output.with_suffix(".stderr").open("xb") as stderr:
    mono_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    start = time.perf_counter_ns()
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, pass_fds=(fd,))
    rc = child.wait()
    end = time.perf_counter_ns()
    mono_end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
with output.open("x") as evidence:
    json.dump(dict(schema="compression-owned-wrapper-wait-v1", argv=argv, child_pid=child.pid,
                   owner_pid=os.getpid(), returncode=rc, started_ns=start, ended_ns=end,
                   monotonic_started_ns=mono_start, monotonic_ended_ns=mono_end,
                   boot_id=boot, same_boot=boot==Path("/proc/sys/kernel/random/boot_id").read_text().strip()), evidence)
    evidence.write("\n")
sys.exit(rc)
