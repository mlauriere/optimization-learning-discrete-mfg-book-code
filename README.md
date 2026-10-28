# Optimization and Learning in Discrete-Time Mean Field Games: Code

Numerical illustrations accompanying *Optimization Frameworks & Learning
Algorithms for Discrete-Time Mean Field Games* by Xin Guo and Mathieu Laurière.
Software authors: Mathieu Laurière and Xin Guo.

This CPU-only package contains a concert-room crowd example, an asymmetric maze
and traffic routing on a Braess network. It includes compact reference data for
replotting and commands for recomputing them. MFGLib-based experiments from the
book are not included. `MODEL_NOTES.md` specifies the implemented models,
algorithm variants and local OpenSpiel adapters.

## Setup

Create a new environment, without overwriting an existing one:

```bash
conda env create --file environment.yml
conda activate discrete-mfg-book
```

On Apple Silicon, use `CONDA_SUBDIR=osx-arm64 conda env create --file environment.yml`
if the conda installation otherwise selects Intel packages. Python 3.10.22 and
OpenSpiel 1.5 are the tested baseline. Direct Python dependencies are pinned in
`requirements.txt`; `requirements-lock.txt` records the tested Python stack,
not a universal platform/build lock. No GPU, Jupyter, PyTorch or TensorFlow is
required. Do not install these dependencies into an unrelated environment.

## Replot the Reference Data

Run from the repository root:

```bash
python verify_reference_data.py
python publication_figures.py --output figures/book
```

The output directory must be new. These commands check the data and produce
nine selected PNG/PDF figure pairs with a figure ledger. They do not import
OpenSpiel or run learning. The ledger identifies the case, algorithm budget,
display choices and timing repetition count. Other saved methods/times can be
displayed using the individual plotters:

```bash
python plot_crowd.py reference_data/maze_c1_p0p5 --output figures/maze-detail --steps 0 2 5 7 10
python plot_traffic.py reference_data/traffic_population --output figures/traffic-detail --figures population --steps 0 1 8 10 16 --columns 3
python plot_traffic_timing.py reference_data/traffic_timing --output figures/timing-detail
```

The manuscript set contains three paired crowd/maze exploitability figures,
four population figures and two traffic figures. Crowd/maze curves display
updates 0--20 and then 25,30,...,1000: the same 217 indices for all 48 curves,
with no smoothing or markers and three-column legends. The full histories
remain available; subsampling changes only the display. Population panels use
the policies returned at update 1000.

All four concert-room cases and both maze cases include eight learning curves,
final-policy flows and final action-probability tables: 48,048 curve values in
total. Traffic includes the four-vehicle population illustration and 60 loop
timings/1,200 update timings from three repetitions at each population. The
data contain configurations, geometry, coordinates, seeds and portable
environment/source provenance. `reference_data/checksums.json` verifies them.
They are plotting/verification data, not resumable optimizer checkpoints.

## Fast Checks

```bash
python -m pytest -q
python crowd.py --config configs/crowd_maze_smoke.json --iterations 2 --horizon 5 --noise 0.5 --methods fp omd_0p1
python traffic.py --config configs/traffic_smoke.json --iterations 2 --timing-iterations 2 --repetitions 1 --populations 2 --horizon 9
```

These are short pipeline checks, not full regeneration. The supplied public
configurations select corrected crowd values and aligned traffic conventions.
Runs create new timestamped directories under `outputs/` containing logs,
resolved settings, package versions, source hashes and numerical data. Replot
them afterwards with the corresponding plotter. Plotting never silently starts
a simulation when data are missing.

## Recompute the Full Examples

```bash
python crowd.py --config configs/crowd_book_baseline.json
python crowd.py --config configs/crowd_maze_baseline.json
python traffic.py --config configs/traffic_aligned_three_repetitions.json --snapshots-only
python traffic_benchmark.py --config configs/traffic_aligned_three_repetitions.json --repetitions 3 --timeout 450
```

The concert room uses H=21, 1000 updates, coefficients 1/10 and noise 0/0.5.
The maze uses H=11, 1000 updates, coefficient 1 and both noise levels. Eight
configurations cover seven algorithm families, including two OMD step sizes.
The traffic population illustration uses N=4 and ten OMD updates. Traffic
timings use H=17, 20 updates per solver and three repetitions for
N=2,5,10,15,20,25,30,40,45,50, with fixed capacity 4 and time step 0.25.

Full runs are not smoke tests. Run timing experiments sequentially without
competing jobs. The recorded complete timing computation, including reusable
pilots, took about 19 minutes with a largest child RSS of 3.25 GiB on the
recorded ARM64/macOS host. Costs on other machines vary; measure a small pilot
before launching the full grid. Numerical library thread counts are limited
to one by the timing controller. Error bars are sample SD, not confidence
intervals. Equal update counts do not mean equal work or equal accuracy.

To replay saved crowd policies with both value formulations, without learning,
use a newly computed crowd run and a new output filename:

```bash
python verify_corrected_crowd.py outputs/<crowd-run> --output outputs/policy-recheck.json
```

Bundled individual-case NPZ files also retain their policy tables; see
`MODEL_NOTES.md` for their interpretation.

## Optional Checkpoints and Resume

Checkpoint saving is off by default. To save restart states during a run:

```bash
python crowd.py --config configs/crowd_maze_smoke.json --iterations 5 --save-checkpoints --checkpoint-interval 1
```

Each method/case writes pickle-free NPZ files under its output directory's
`checkpoints/METHOD/` subdirectory. Resume one saved method/case into a new run:

```bash
python crowd.py --resume outputs/<run>/<case>/checkpoints/<method>/update_000005.npz --iterations 10 --save-checkpoints
```

`--iterations` is the desired total, not the number of additional updates.
The model and hyperparameters are retained; incompatible versions, core source
hashes and requested model changes are rejected. Use the recorded environment
and saved `sources/` files to resume an older implementation. This currently
supports only the corrected normalized crowd backend. `--no-save-checkpoints`
disables further saves; the default interval is 25, with initial/final saves.
Fine checkpoint intervals can consume substantial space on longer runs.

The bundled 1000-update histories were extended from saved solver states,
retaining all earlier values. Large restart archives are not distributed.
Bundled final-policy tables permit evaluation and replotting, not continuation
of the optimizer; create checkpoints during your own runs to enable resume.
New runs also embed their plot metadata in `results.npz` and save source and
environment records separately.

## Interpretation and Repeatability

All public reference data use the local corrections described in
`MODEL_NOTES.md`; earlier native-library diagnostic runs are not mixed in.
The corrected crowd policies were checked with two value formulations, and the
traffic adapters have pathwise finite/empirical-mean-field regression tests.
These checks are specific to the pinned configurations, not certification of
every OpenSpiel game or routine. No parameters were tuned to fit historical
plots. The figures illustrate finite learning budgets, not universal convergence
or algorithm rankings. Some final policies remain highly exploitable.

Use the recorded packages and configurations for close numerical agreement.
Hardware-independent runtimes, byte-identical PDFs and compatibility with
other OpenSpiel versions are not promised. The source hashes in data provenance
refer to the numerical runs before packaging; configuration defaults and the
publication ledger have been adapted for this standalone repository.

## License and Citation

Original code and accompanying numerical reference outputs are distributed
under the MIT License. Dependencies retain their own licenses; see
`THIRD_PARTY_NOTICES.md`. `CITATION.cff` records software authorship. Please cite
the accompanying book when using these examples.
