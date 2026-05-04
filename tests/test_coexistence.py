"""
Coexistence test: run nuclei segmentation with both cellpose3_legacy (CP3)
and cellpose >= 4 (cellpose-SAM / CP4) on the same image and compare results.

Run with the sdcpsam conda environment:
    conda activate sdcpsam
    python tests/test_coexistence.py
or:
    pytest tests/test_coexistence.py -v
"""

import warnings
import numpy as np
from pathlib import Path
import torch

DATA_DIR = Path(__file__).parent / "data"
IMAGE_PATH = DATA_DIR / "nuclei_large.tif"

GPU = torch.cuda.is_available()


def load_image():
    from cellpose3_legacy import io
    img = io.imread(str(IMAGE_PATH))
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    return img


def segment_with_cp3(img):
    """Segment using cellpose3_legacy (CP3 / cellpose <4)."""
    from cellpose3_legacy import models as models3
    model = models3.CellposeModel(gpu=GPU, pretrained_model="nuclei")
    with warnings.catch_warnings():
        # Suppress PyTorch sparse tensor invariant warning (harmless on newer torch)
        warnings.filterwarnings(
            "ignore",
            message="Sparse invariant checks are implicitly disabled",
            category=UserWarning,
        )
        masks, flows, styles = model.eval(
            img,
            diameter=None,       # auto-estimate
            channels=[0, 0],     # grayscale
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )
    return masks


def segment_with_cp4(img):
    """Segment using cellpose >= 4 (cellpose-SAM).
    CP4 uses a single universal model (cpsam) rather than per-type models.
    """
    from cellpose import models as models4
    model = models4.CellposeModel(gpu=GPU, pretrained_model="cpsam")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="channels deprecated", category=UserWarning)
        masks, flows, styles = model.eval(
            img,
            diameter=None,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )
    return masks


def report(name, masks):
    n_cells = masks.max()
    print(f"  [{name}] detected cells: {n_cells}")
    return n_cells


def test_coexistence():
    assert IMAGE_PATH.exists(), f"Test image not found: {IMAGE_PATH}"
    print(f"GPU: {'enabled (' + torch.cuda.get_device_name(0) + ')' if GPU else 'not available, using CPU'}")

    img = load_image()

    print("\n--- cellpose3_legacy (CP3) ---")
    masks_cp3 = segment_with_cp3(img)
    n_cp3 = report("CP3", masks_cp3)

    print("\n--- cellpose >= 4 (CP4 / SAM) ---")
    masks_cp4 = segment_with_cp4(img)
    n_cp4 = report("CP4", masks_cp4)

    # Both models should find at least one cell
    assert n_cp3 > 0, "cellpose3_legacy found no cells"
    assert n_cp4 > 0, "cellpose >=4 found no cells"

    # Results may differ between versions, but should be in the same ballpark
    # (within 50% of each other) — purely a sanity check, not a strict comparison
    ratio = min(n_cp3, n_cp4) / max(n_cp3, n_cp4)
    assert ratio >= 0.50, (
        f"Cell counts differ too much between versions: CP3={n_cp3}, CP4={n_cp4}"
    )

    rows = [
        ("cellpose3_legacy", "nuclei  (CP3)",     n_cp3),
        ("cellpose ≥4",      "cpsam  (CP4/SAM)",  n_cp4),
    ]
    print(f"\n{'Algorithm':<22}{'Model':<22}{'Cells':>6}  {'Agreement':>10}")
    print("-" * 64)
    for algo, model, n in rows:
        r = min(n, max(n_cp3, n_cp4)) / max(n, max(n_cp3, n_cp4))
        print(f"{algo:<22}{model:<22}{n:>6}  {r:>9.2f}")
    print(f"\nAgreement ratio: {ratio:.2f} (threshold: 0.50)")
    print("\nPASS: both versions coexist and produce comparable results.")


if __name__ == "__main__":
    test_coexistence()
