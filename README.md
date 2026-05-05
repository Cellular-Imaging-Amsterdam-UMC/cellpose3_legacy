# cellpose3_legacy

> **A renamed legacy fork of the Cellpose CP3 branch, intended to coexist with Cellpose 4 / Cellpose-SAM (and cistardist_pytorch) in the same Python environment.**

Original upstream: [MouseLand/cellpose](https://github.com/MouseLand/cellpose)

---

## Installation

### From GitHub (recommended)

```bash
pip install --no-cache-dir \
    "git+https://github.com/Cellular-Imaging-Amsterdam-UMC/cellpose3_legacy@cp3"
```

### In a local virtual environment

```bash
# Create and activate a new environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac / Linux
source .venv/bin/activate

# Install
pip install --no-cache-dir \
    "git+https://github.com/Cellular-Imaging-Amsterdam-UMC/cellpose3_legacy@cp3"
```

### With conda

```bash
conda create -n cellpose3_legacy python=3.11
conda activate cellpose3_legacy
pip install --no-cache-dir \
    "git+https://github.com/Cellular-Imaging-Amsterdam-UMC/cellpose3_legacy@cp3"
```

### From a local clone (editable)

```bash
git clone https://github.com/Cellular-Imaging-Amsterdam-UMC/cellpose3_legacy.git
cd cellpose3_legacy
pip install -e .
```

### Coexistence with Cellpose >= 4

To use both Cellpose <=3 and Cellpose >=4 side by side:

```bash
pip install cellpose          # installs cellpose 4 / cellpose-SAM
pip install --no-cache-dir \
    "git+https://github.com/Cellular-Imaging-Amsterdam-UMC/cellpose3_legacy@cp3"
```

```python
import cellpose               # cellpose 4 / SAM
import cellpose3_legacy       # cellpose 3 (this fork)

from cellpose3_legacy import models, io
```

---

## Coexistence tests

Two ready-to-run tests in `tests/` validate all three algorithms in a single PyTorch environment:

| Test file | Algorithms |
|---|---|
| [`tests/test_coexistence.py`](tests/test_coexistence.py) | cellpose3_legacy (CP3) vs cellpose >=4 (CP4/SAM) |
| [`tests/test_coexistence_cp3_cp4_sd.py`](tests/test_coexistence_cp3_cp4_sd.py) | CP3 · CP4/SAM · StarDist 2D — with CPU & GPU timing and label montage |

The three-algorithm test uses [cistardist-pytorch](https://pypi.org/project/cistardist-pytorch/)
([GitHub](https://github.com/Cellular-Imaging-Amsterdam-UMC/cistardist_pytorch)),
a PyTorch-only StarDist 2D implementation with no TensorFlow/Keras dependency.

```bash
pip install cistardist-pytorch
pytest tests/test_coexistence_cp3_cp4_sd.py -v
```

The test runs each algorithm on CPU and GPU, prints a timing table, and saves a
coloured label montage to `tests/output/montage_cp3_cp4_sd.png`.

![CP3 · CP4/SAM · StarDist 2D label montage](images/montage_cp3_cp4_sd_2008x2008.png)

```
Algorithm             Model                      Cells   CPU (s)   GPU (s)
----------------------------------------------------------------------------
cellpose3_legacy      nuclei  (CP3)               1100     25.54      4.85
cellpose >=4          cpsam  (CP4/SAM)            1137    257.88      5.35
StarDist 2D           SD_Nuclei_Versatile         1118      2.53      1.68
```

---

## Citation

If you use this package, please cite the original Cellpose papers:

**Cellpose 1.0:**
Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods, 18*(1), 100-106. [paper](https://www.nature.com/articles/s41592-020-01018-x)

**Cellpose 2.0:**
Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: how to train your own model. *Nature Methods*. [paper](https://www.nature.com/articles/s41592-022-01663-4)

**Cellpose 3.0:**
Stringer, C. & Pachitariu, M. (2025). Cellpose3: one-click image restoration for improved segmentation. *Nature Methods*. [paper](https://www.nature.com/articles/s41592-025-02595-5)