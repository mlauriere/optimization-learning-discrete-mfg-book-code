"""Publication-oriented style and provenance for data-only plot commands."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import os
from pathlib import Path

from common import ROOT, write_json

# Solver commands do not import this module or initialize Matplotlib.
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_figure(fig, directory, name):
    for suffix in ("png", "pdf"):
        fig.savefig(Path(directory) / f"{name}.{suffix}", dpi=160,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def paper_style(scale=1.0):
    if not 0.5 <= scale <= 3:
        raise ValueError("Font scale must lie in [0.5,3]")
    return plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 16 * scale,
        "axes.labelsize": 19 * scale, "axes.titlesize": 18 * scale,
        "xtick.labelsize": 16 * scale, "ytick.labelsize": 16 * scale,
        "legend.fontsize": 14 * scale, "lines.linewidth": 2.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        # Avoid incorrect low-bit-depth palette predictors in the pinned PDF backend.
        "pdf.compression": 0,
    })


def plot_directory(source, output, options, input_files):
    """Create a fresh figure directory and record inputs and rendering options."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    source = Path(source).resolve()
    directory = Path(output) if output else source / "figures" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    files = sorted({Path(path).resolve() for path in input_files})
    write_json(directory / "plot_manifest.json", {
        "schema_version": 1, "created_utc": stamp, "source": str(source),
        "options": options,
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in files},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(ROOT.glob("*.py"))},
        "packages": {name: version(name) for name in ("numpy", "matplotlib")},
        "status": "rendering",
    })
    return directory


def finish_plots(directory, details=None):
    from common import read_config
    path = Path(directory) / "plot_manifest.json"
    manifest = read_config(path)
    if details:
        manifest.update(details)
    manifest["status"] = "complete"
    manifest["figures"] = sorted(str(p.relative_to(directory))
                                 for p in Path(directory).rglob("*")
                                 if p.suffix in (".png", ".pdf"))
    write_json(path, manifest)
