# FloodMapper

We present research to detect flooding changes in NASA’s Uninhabited Aerial Vehicle Synthetic Aperture Radar (UAVSAR) imagery using deep learning, and to classify the land cover types affected by these floods. First, a binary change detection model is used to identify flooded areas by comparing pre-event optical data and post-event UAVSAR data. Next, the regions are classified by land cover category to determine what types of terrain experienced the most inundation. The outcome is an integrated flood mapping and land cover analysis tool, which is important for decisions regarding where priority disaster response is dispatched and for land use analysis. This tool provides timely flood extent maps regardless of weather or daylight conditions and helps to understand which land covers and communities are impacted by flooding to aid in recovery efforts, where to develop preventative infrastructure, and future urban planning. The work builds on state-of-the-art deep learning methods for image segmentation and leverage geospatial data platforms to efficiently handle large remote sensing datasets. We train and test the models on UAVSAR data, and the models achieve 90.0\%, pixel accuracy for change detection and 86.1\% pixel accuracy for land cover classification. 

## Download

[![Releases](https://img.shields.io/github/v/release/kevintang6142/flood-detection?include_prereleases&sort=semver)](https://github.com/kevintang6142/flood-detection/releases)
[![Downloads](https://img.shields.io/github/downloads/kevintang6142/flood-detection/total?label=downloads)](https://github.com/kevintang6142/flood-detection/releases)

## Overview

This repository contains a research prototype for mapping flooded areas by combining SAR imagery with optical imagery, and classifying land cover within the affected regions. It includes:

- A Qt-based desktop UI for interactive analysis (`app/frontend.py`)
- A command-line backend pipeline (`app/backend.py`) for tiling, filtering, classification, and stitching outputs
- Example data folders and scripts used during experimentation

The pipeline:
1) Reproject SAR to a meter-based CRS and apply a Refined Lee filter.
2) Tile SAR and optical images to fixed-size PNGs for model input.
3) Run land cover segmentation using SegFormer with custom weights.
4) Perform change detection over pre-/post-event imagery.
5) Combine and stitch tiles; compute summary statistics and render a legend.

## Repository structure

- `app/`
	- `frontend.py` — PySide6 (Qt) GUI application
	- `backend.py` — CLI pipeline and helper utilities
	- `SegformerJaccardLoss.pth` — SegFormer weights expected by the frontend
	- `FloodMapperWindows.spec`, `FloodMapperMac.spec` — PyInstaller specs (optional packaging)
- `new_data_cd/`, `new_data_lcc/`, `tiles/` — data and tile directories used during experiments
- `scripts/` — notebooks and helpers
- `pyproject.toml`, `requirements.txt`, `uv.lock` — Python project and dependency manifests

## Requirements

- Python 3.12+
- OS: Windows 10/11 or macOS 12+
- GPU is optional; Torch will fall back to CPU if CUDA is unavailable.

Python dependencies: rasterio, numpy, scipy, Pillow, OpenCV, PySide6, torch, torchvision, segmentation-models-pytorch, matplotlib.

System notes for rasterio/GDAL:
- Windows: official wheels typically bundle GDAL; installation via `uv` or `pip` should work out-of-the-box.
- macOS: if you hit GDAL-related build/runtime issues, install GDAL with Homebrew first: `brew install gdal`.

## Setup

### 1) Get the code (clone and cd)

```powershell
# Windows PowerShell
git clone https://github.com/kevintang6142/flood-detection.git
cd .\flood-detection
```

```bash
# macOS / Linux
git clone https://github.com/kevintang6142/flood-detection.git
cd flood-detection
```

### 2) Install dependencies

You can use `uv` (recommended) or `pip`.

#### Option A: Using uv

```powershell
uv sync
```

#### Option B: Using pip and a virtual environment

```powershell
# Windows PowerShell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the frontend (GUI)

The GUI expects the weights file `SegformerJaccardLoss.pth` to be in the `app/` folder. From the project root:

```powershell
# Windows PowerShell
uv run python app\frontend.py
```

```bash
# macOS / Linux
uv run python app/frontend.py
```

If using pip/venv, replace `uv run` with `python` (or `python3` on macOS/Linux).

In the app, select:
- SAR TIF: path to the SAR (flight path) GeoTIFF
- Optical TIF: path to the optical GeoTIFF

Then click "Run Analysis". The GUI logs progress, shows the final flood map, and renders a legend. You can save both images via "Save Results…".

## Running the backend (CLI)

Command entry point: `uv run app/backend.py`

If using pip/venv, replace `uv run` with `python` (or `python3` on macOS/Linux).

Positional arguments (required):
- `sar_tif` (path): Path to the SAR (flight path) GeoTIFF.
- `optical_tif` (path): Path to the optical GeoTIFF for the same area.
- `lc_weights_file` (path): Path to the land cover model weights file (e.g., app/SegformerJaccardLoss.pth).
- `gan_weights_file` (path): Path to the CycleGAN model weights file (e.g., app/ganmodel.pth).
- `cd_weights_file` (path): Path to the change detection model weights file (e.g., app/changemodel.pt).

Optional arguments:
- `--output-dir OUTPUT_DIR` (path): Directory where outputs will be saved. If provided, two PNGs are written: a flood map and a legend. If omitted, the CLI will display the images.
- Output filenames are derived from the SAR filename: `<sar_base>_flood_map.png` and `<sar_base>_legend.png`.


## Packaging

Prebuilt artifacts (when available) are published on the Releases page.

If you wish to build a standalone executable yourself, PyInstaller spec files are included:

```powershell
# Windows PowerShell
cd app
uv run pyinstaller app\FloodMapperWindows.spec
```

```bash
# macOS / Linux
cd app
uv run pyinstaller app/FloodMapperMacOS.spec
```

If using pip/venv, remove `uv run`.

Artifacts will be created under `app/build/` and `app/dist/`.

## Acknowledgements

The change detection model integrated into this project is based on the work from the [ChangeDetection-GLCD-DA](https://github.com/WHU-SGG-RS-Pro-Group/ChangeDetection-GLCD-DA) repository. We thank the authors for making their code publicly available.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
