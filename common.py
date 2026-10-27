"""Run records and numerical-data helpers shared by the two examples."""

import csv
from datetime import datetime
from importlib.metadata import distributions, version
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
import numpy as np


def read_config(path):
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path, value):
    with Path(path).open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("No records to save")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_mass(values, axis, tolerance=1e-6):
    values = np.asarray(values)
    if not np.isfinite(values).all() or values.min() < -tolerance:
        raise ValueError("Distribution contains nonfinite or negative masses")
    error = float(np.max(np.abs(values.sum(axis=axis) - 1)))
    if error > tolerance:
        raise ValueError(f"Distribution mass error {error} exceeds {tolerance}")
    return error


class Run:
    """Own a unique output directory and record success or failure on exit."""
    def __init__(self, example, config, output_root=None):
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        self.path = Path(output_root or ROOT / "outputs") / f"{example}_{stamp}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.log_handle = (self.path / "run.log").open("w")
        write_json(self.path / "config.json", config)
        self.manifest = {
            "example": example,
            "status": "running",
            "started_asia_shanghai": stamp,
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "thread_limits": {key: os.environ.get(key) for key in
                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "packages": {name: version(name) for name in
                         ("open-spiel", "numpy", "scipy", "matplotlib",
                          "pandas", "seaborn")},
            "claim": "Numerical illustration with the recorded budget; not exact historical reproduction",
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(ROOT.glob("*.py"))},
        }
        lock = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())
        (self.path / "requirements-lock.txt").write_text("\n".join(lock) + "\n")
        write_json(self.path / "manifest.json", self.manifest)

    def log(self, stage, message):
        line = f"[elapsed={time.perf_counter() - self.started:9.3f}s] [{stage}] {message}"
        print(line, flush=True)
        self.log_handle.write(line + "\n")
        self.log_handle.flush()

    def __enter__(self):
        self.log("setup", f"output_dir={self.path}")
        self.log("setup", f"python={sys.executable} machine={platform.machine()}")
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.manifest["elapsed_seconds"] = time.perf_counter() - self.started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.manifest["peak_rss_mb"] = peak / (1024**2 if sys.platform == "darwin" else 1024)
        self.manifest["status"] = "failed" if exc_type else "complete"
        if exc_type:
            self.manifest["error"] = str(exc_value)
            self.log("failed", "".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        else:
            self.log("done", f"complete; peak_rss_mb={self.manifest['peak_rss_mb']:.1f}")
        write_json(self.path / "manifest.json", self.manifest)
        self.log_handle.close()
        return False
