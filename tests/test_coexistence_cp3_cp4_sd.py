"""
Coexistence test: nuclei segmentation with three algorithms in one PyTorch env.

  1. cellpose3_legacy  — CP3 / cellpose <4  (model: nuclei)
  2. cellpose >=4      — CP4 / cellpose-SAM  (model: cpsam)
  3. cistardist_pytorch — StarDist 2D        (model: SD_Nuclei_Versatile)

Each algorithm is timed on CPU and then on GPU (if available). Only the GPU
label images are shown in the montage. Both CPU and GPU inference times are
printed to the console and displayed above each label panel.

Timing covers the model.eval() / predict_instances() call only (model loading
is excluded).

Run with the sdcpsam conda environment:
    conda activate sdcpsam
    python tests/test_coexistence_cp3_cp4_sd.py
or:
    pytest tests/test_coexistence_cp3_cp4_sd.py -v
"""

import argparse
import time
import warnings
import numpy as np
from pathlib import Path
import torch
import tifffile
import matplotlib
matplotlib.use("Agg")            # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TESTS_DIR    = Path(__file__).parent
DATA_DIR     = TESTS_DIR / "data"
SD_MODEL_DIR = TESTS_DIR / "SD_Nuclei_Versatile"
IMAGE_PATH   = DATA_DIR / "nuclei_large.tif"
OUTPUT_DIR   = TESTS_DIR / "output"

HAS_GPU = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _sync():
    """Flush CUDA queue so wall-clock time is accurate."""
    if HAS_GPU:
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------

def load_image() -> np.ndarray:
    img = tifffile.imread(str(IMAGE_PATH))
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    return img


# ---------------------------------------------------------------------------
# Colourmap
# ---------------------------------------------------------------------------

def _random_label_cmap(n: int = 256, seed: int = 42) -> ListedColormap:
    """Perceptually distinct colour per label; 0 → black background."""
    rng = np.random.default_rng(seed)
    colors = np.ones((n, 4))
    colors[0] = [0, 0, 0, 1]
    hsv = np.stack([
        rng.uniform(0, 1, n - 1),
        rng.uniform(0.5, 1, n - 1),
        rng.uniform(0.6, 1, n - 1),
    ], axis=1)
    colors[1:, :3] = mcolors.hsv_to_rgb(hsv)
    colors[1:, 3] = 1.0
    return ListedColormap(colors)


# ---------------------------------------------------------------------------
# Montage
# ---------------------------------------------------------------------------

def save_montage(
    results: list[tuple[str, str, np.ndarray, float, float]],
    raw: np.ndarray,
):
    """Save one-row montage.

    ``results`` items: (algo_name, model_name, gpu_masks, cpu_sec, gpu_sec)
    ``gpu_sec`` is None when no GPU is available.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmap = _random_label_cmap()

    n_panels = 1 + len(results)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6),
                             constrained_layout=True)
    fig.patch.set_facecolor("#1a1a1a")

    # Raw image panel
    ax = axes[0]
    ax.imshow(raw, cmap="gray", interpolation="nearest")
    ax.set_title("Input image\nnuclei.tif", color="white", fontsize=10, pad=6)
    ax.axis("off")

    # Label panels
    for ax, (algo_name, model_name, masks, cpu_sec, gpu_sec) in zip(axes[1:], results):
        n_cells = int(masks.max())

        # Build per-image colourmap (cycle for images with >256 labels)
        lmap = ListedColormap([cmap(i % 256) for i in range(n_cells + 1)])
        lmap.colors[0] = [0, 0, 0, 1]
        ax.imshow(masks, cmap=lmap, interpolation="nearest",
                  vmin=0, vmax=max(n_cells, 1))

        # Title: name + model + cells on first two lines, timings on third
        if cpu_sec is not None:
            timing_line = f"CPU {cpu_sec:.2f} s"
            if gpu_sec is not None:
                timing_line += f"  |  GPU {gpu_sec:.2f} s"
        elif gpu_sec is not None:
            timing_line = f"GPU {gpu_sec:.2f} s"
        else:
            timing_line = ""
        title = f"{algo_name}\n{model_name}  ·  {n_cells} cells\n{timing_line}"
        ax.set_title(title, color="white", fontsize=9, pad=6, linespacing=1.5)
        ax.axis("off")

    out_path = OUTPUT_DIR / "montage_cp3_cp4_sd.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nMontage saved → {out_path}")


# ---------------------------------------------------------------------------
# Segmentation functions  (gpu=True/False controls device)
# ---------------------------------------------------------------------------

def segment_cp3(img: np.ndarray, gpu: bool) -> tuple[np.ndarray, float]:
    """cellpose3_legacy — CP3 nuclei model. Returns (masks, elapsed_s)."""
    from cellpose3_legacy import models as models3
    model = models3.CellposeModel(gpu=gpu, pretrained_model="nuclei")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse invariant checks are implicitly disabled",
            category=UserWarning,
        )
        _sync()
        t0 = time.perf_counter()
        masks, _, _ = model.eval(
            img, diameter=None, channels=[0, 0],
            flow_threshold=0.4, cellprob_threshold=0.0,
        )
        _sync()
        elapsed = time.perf_counter() - t0
    return masks.astype(np.int32), elapsed


def segment_cp4(img: np.ndarray, gpu: bool) -> tuple[np.ndarray, float]:
    """cellpose >=4 — universal cpsam model. Returns (masks, elapsed_s)."""
    from cellpose import models as models4
    model = models4.CellposeModel(gpu=gpu, pretrained_model="cpsam")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="channels deprecated",
                                category=UserWarning)
        _sync()
        t0 = time.perf_counter()
        masks, _, _ = model.eval(
            img, diameter=None,
            flow_threshold=0.4, cellprob_threshold=0.0,
        )
        _sync()
        elapsed = time.perf_counter() - t0
    return masks.astype(np.int32), elapsed


def segment_sd(img: np.ndarray, gpu: bool) -> tuple[np.ndarray, float]:
    """cistardist_pytorch — StarDist2D SD_Nuclei_Versatile. Returns (masks, elapsed_s)."""
    from cistardist_pytorch import StarDist2D
    device = "cuda" if gpu else "cpu"
    model = StarDist2D.from_folder(str(SD_MODEL_DIR), device=device)
    _sync()
    t0 = time.perf_counter()
    labels, _ = model.predict_instances(img, normalize=True)
    _sync()
    elapsed = time.perf_counter() - t0
    return labels.astype(np.int32), elapsed


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

# Algorithm registry: (display_name, model_label, segment_fn)
ALGORITHMS = [
    ("cellpose3_legacy", "nuclei  (CP3)",       segment_cp3),
    ("cellpose ≥4",      "cpsam  (CP4/SAM)",    segment_cp4),
    ("StarDist 2D",      "SD_Nuclei_Versatile", segment_sd),
]


def _run_all(img: np.ndarray, gpu: bool, label: str) -> list[tuple]:
    """Run every algorithm with ``gpu`` flag. Returns list of (name, model, masks, elapsed)."""
    tag = "GPU" if gpu else "CPU"
    print(f"\n{'='*50}")
    print(f"  {label}  [{tag}]")
    print(f"{'='*50}")
    run_results = []
    for algo_name, model_name, fn in ALGORITHMS:
        print(f"\n  {algo_name} ({model_name}) …", end="", flush=True)
        masks, elapsed = fn(img, gpu=gpu)
        n = int(masks.max())
        print(f"  {n} cells  |  {elapsed:.2f} s")
        run_results.append((algo_name, model_name, masks, elapsed))
    return run_results


def test_coexistence_cp3_cp4_sd(gpu_only: bool = False):
    assert IMAGE_PATH.exists(), f"Test image not found: {IMAGE_PATH}"
    assert SD_MODEL_DIR.exists(), f"SD model dir not found: {SD_MODEL_DIR}"

    if gpu_only and not HAS_GPU:
        raise RuntimeError("--gpu_only requested but no CUDA GPU is available")

    device_info = (
        f"GPU enabled  ({torch.cuda.get_device_name(0)})" if HAS_GPU
        else "GPU not available — CPU only"
    )
    print(f"\n{device_info}")
    img = load_image()

    # ---- CPU run (skipped when --gpu_only) ----
    if gpu_only:
        print("\n[--gpu_only] skipping CPU run")
        cpu_runs = None
    else:
        cpu_runs = _run_all(img, gpu=False, label="CPU run")

    # ---- GPU run (if available) ----
    if HAS_GPU:
        gpu_runs = _run_all(img, gpu=True, label="GPU run")
    else:
        gpu_runs = cpu_runs  # same data; no separate GPU pass

    # ---- Collect montage data ----
    montage_data = []
    for i, (algo, model, gpu_masks, gpu_t) in enumerate(gpu_runs):
        cpu_t = cpu_runs[i][3] if cpu_runs is not None else None
        montage_data.append((
            algo, model, gpu_masks,
            cpu_t,
            gpu_t if HAS_GPU else None,
        ))

    # ---- Assertions ----
    for algo, model, masks, cpu_t, gpu_t in montage_data:
        n = int(masks.max())
        assert n > 0, f"{algo} found no cells"

    counts = [int(m[2].max()) for m in montage_data]
    median = np.median(counts)
    for (algo, _, _, _, _), n in zip(montage_data, counts):
        ratio = min(n, median) / max(n, median)
        assert ratio >= 0.40, (
            f"{algo} cell count ({n}) deviates too much from median ({median:.0f})"
        )

    save_montage(montage_data, img)

    # ---- Summary table ----
    print(f"\n{'Algorithm':<22}{'Model':<26}{'Cells':>6}  {'CPU (s)':>8}  {'GPU (s)':>8}")
    print("-" * 76)
    for algo, model, masks, cpu_t, gpu_t in montage_data:
        n = int(masks.max())
        cpu_str = f"{cpu_t:8.2f}" if cpu_t is not None else "     n/a"
        gpu_str = f"{gpu_t:8.2f}" if gpu_t is not None else "     n/a"
        print(f"{algo:<22}{model:<26}{n:>6}  {cpu_str}  {gpu_str}")
    print("\nPASS: all three algorithms coexist and produce comparable results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coexistence test: CP3 / CP4 / StarDist")
    parser.add_argument(
        "--gpu_only",
        action="store_true",
        help="Skip CPU inference and run GPU inference only",
    )
    args = parser.parse_args()
    test_coexistence_cp3_cp4_sd(gpu_only=args.gpu_only)
