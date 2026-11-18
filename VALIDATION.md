# Release validation — 2026-11-18

This record describes the Windows Conda source edition 0.0.1, based on the current
TwoPhoton Studio 1.3.4 application. Tests used synthetic data, not experimental datasets.

## Environment

- Windows x64 workstation; NVIDIA GeForce RTX 4060, compute capability 8.9.
- Two newly created, isolated conda-forge prefixes: Python 3.10.22 (GUI) and
  Python 3.11.18 (models). Existing analysis environments were not modified.
- Environment creation used the official micromamba executable through the installer's
  Conda-compatible create command. A custom --env-root was used for local validation.
- GUI: Suite2p 0.10.3, NumPy 1.24.4, PyTorch 2.4.1+cpu, PyQt5 5.15.9.
- Models: Suite2p 1.0.0.1, Cellpose 4.0.8, PyTorch 2.7.0+cu128.
- The full tested Python dependency versions are in requirements/*-constraints.txt.
- MATLAB R2024b was already installed and was found automatically.

## Passed checks

| Check | Evidence / scope |
| --- | --- |
| Installer and asset tests | All six unittest checks passed; bundled SHA256/size checks passed |
| Source syntax | compileall passed for application, scripts and installer |
| Git checkout integrity | All six tests also passed on a fresh Git-index checkout; bundled SHA256 values survived checkout |
| Dependency consistency | pip check passed in both new environments |
| GUI construction | Offscreen Studio construction, default tau and Fit FOV button checked |
| Installed command entry point | cli.main opened the GUI; its spawned Windows worker imported a synthetic two-frame TIFF |
| Multi-stack ordering | Two TIFFs, 120 frames each; natural order and frame-to-file mapping checked |
| Fluorescence extraction | F and Fneu compared numerically against the expected pixel means |
| Deconvolution | OASIS output and independent-session boundary behavior checked |
| ROI editing/extraction | Duplicate rejection and manual addition followed by recomputation checked |
| Functional detection | Native legacy Suite2p detected three synthetic ROIs |
| MAT export | Fall.mat read back with the expected fluorescence and frame map |
| CPU registration | Legacy Suite2p nonrigid registration completed on 120 synthetic frames |
| CUDA execution | Real GPU matrix multiplication, not only is_available() |
| GPU extraction | Modern Suite2p extraction returned the expected array dimensions |
| GPU registration | Modern Suite2p nonrigid registration completed on 60 synthetic frames |
| PROFILE | Bundled m124 checkpoint loaded; finite vector field produced |
| SUPPORT | Bundled bs1 checkpoint loaded; patch stitching produced a finite two-frame movie |
| Cellpose-SAM | Official checkpoint SHA256 verified; model inference completed on a small image |
| MATLAB FIG | Application heatmap exporter created a FIG; MATLAB openfig reopened it with saved Visible=on |
| Installer completion | ALL REQUIRED CHECKS PASSED and INSTALLATION COMPLETE were reached |

The MATLAB FIG check exercised the release exporter using the workstation's existing
GUI interpreter and installed MATLAB. The new GUI environment separately passed MAT
export and application/worker checks.

The installer was rerun after correcting dependency constraints and an incorrect local
test MATLAB path. It reused its owned environments and verified checkpoint. Explicit
MATLAB paths are now validated before environment installation starts. No MATLAB Runtime
installer is included or invoked.

## Reproduce locally

From the repository, after installation:

```bat
python install.py --check
conda activate <your-gui-environment-path>
python -m unittest discover -s tests -v
python scripts/smoke_pipeline.py .local/smoke-new
python scripts/smoke_launch.py
```

Use a new output directory for the pipeline smoke test; it intentionally refuses
to overwrite exported results. scripts/doctor.py is the installer's model/GUI check.

The local installation keeps its reports in .local/doctor-report.json,
.local/core-installed.txt, .local/models-installed.txt and .local/install.log.
Those reports may contain local paths and are excluded from the source archive.
The GitHub workflow runs source checks on pushes and offers an optional full Windows CPU
installation. Live workflow results are available on the repository's
[Actions page](https://github.com/yjlu2/TwoPhotonStudio/actions).

## Limits

- This was a new-environment test on an existing Windows workstation, not a clean OS VM.
- No independent end-to-end installation of the CPU-only PyTorch wheels was performed.
  Native CPU registration/extraction/deconvolution were tested in the new GUI environment.
- Other GPUs (including RTX 50-series), Linux and macOS were not tested.
  Each CUDA installation runs its own real kernel and model checks.
- Small synthetic tests verify execution and selected numerical invariants. They do not
  establish biological segmentation accuracy, registration quality for all motion patterns,
  or performance/memory use on large experiments.
- Direct and transitive Python versions are constrained; Conda runtime packages/Python
  patch releases are not fully hash-locked.

