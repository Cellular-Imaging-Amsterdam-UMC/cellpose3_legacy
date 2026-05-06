"""
Profile CP3 and cistardist_pytorch StarDist inference stage by stage.

This is a benchmark utility, not a pass/fail unit test. It mirrors the
cistardist_pytorch profiling style and exposes where CP3 spends time:
normalization, network/tiling, flow following, mask creation, flow QC, cleanup,
and flow-display packaging.

Example:
    python tests/profile_cp3_stardist_speed.py --gpu_only
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tifffile
import torch


TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"
SD_MODEL_DIR = TESTS_DIR / "SD_Nuclei_Versatile"
IMAGE_PATH = DATA_DIR / "nuclei_large.tif"


@dataclass
class ProfileResult:
    name: str
    device: str
    timings: dict[str, float]
    metadata: dict[str, Any]
    masks: np.ndarray | None = None


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(
    timings: dict[str, float],
    name: str,
    device: torch.device,
    func: Callable[[], Any],
) -> Any:
    synchronize(device)
    start = time.perf_counter()
    value = func()
    synchronize(device)
    timings[name] = time.perf_counter() - start
    return value


def format_seconds(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:8.1f} us"
    if seconds < 1:
        return f"{seconds * 1000:8.2f} ms"
    return f"{seconds:8.3f} s "


def profile_cp3(
    image: np.ndarray,
    device_name: str,
    name: str,
    fast_mode: bool = False,
    niter: int | None = None,
    interp: bool = True,
    resample: bool = True,
    flow_threshold: float = 0.4,
    diameter: float | None = None,
    cellprob_threshold: float = 0.0,
) -> ProfileResult:
    import cv2
    from cellpose3_legacy import dynamics, models as models3, plot, transforms, utils

    model = models3.CellposeModel(gpu=device_name.startswith("cuda"), pretrained_model="nuclei")
    device = model.device
    timings: dict[str, float] = {}
    metadata: dict[str, Any] = {
        "image_shape": tuple(int(v) for v in image.shape),
        "image_dtype": str(image.dtype),
        "fast_mode": fast_mode,
    }

    if fast_mode:
        diameter = 17.0 if diameter is None else diameter
        resample = False
        interp = True
        flow_threshold = 0.4
        niter = 100 if niter is None else niter

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse invariant checks are implicitly disabled",
            category=UserWarning,
        )
        total_start = time.perf_counter()

        x = timed(
            timings,
            "convert_image",
            device,
            lambda: transforms.convert_image(image, [0, 0], nchan=model.nchan),
        )
        if x.ndim < 4:
            x = x[np.newaxis, ...]

        if diameter is not None and diameter > 0:
            rescale = model.diam_mean / diameter
        else:
            rescale = model.diam_mean / model.diam_labels
        metadata["diameter"] = diameter
        metadata["rescale"] = float(rescale)
        metadata["resample"] = resample
        metadata["interp"] = interp
        metadata["flow_threshold"] = flow_threshold
        metadata["cellprob_threshold"] = cellprob_threshold

        normalize_params = dict(models3.normalize_default)
        normalize_params["normalize"] = True
        normalize_params["invert"] = False
        x = timed(
            timings,
            "normalize",
            device,
            lambda: transforms.normalize_img(np.asarray(x), **normalize_params),
        )

        dP, cellprob, styles = timed(
            timings,
            "network_tiling_forward",
            device,
            lambda: model._run_net(
                x,
                rescale=rescale,
                augment=False,
                batch_size=8,
                tile_overlap=0.1,
                bsize=224,
                resample=resample,
                do_3D=False,
                anisotropy=None,
            ),
        )
        del styles

        niter_resolved = 200 if not resample else (1 / rescale * 200)
        niter_resolved = niter_resolved if niter is None or niter == 0 else niter
        niter_resolved = int(niter_resolved)
        metadata["niter"] = niter_resolved
        metadata["network_shape"] = tuple(int(v) for v in cellprob[0].shape)

        cellprob_i = cellprob[0]
        dP_i = dP[:, 0]
        resize = None if cellprob_i.shape == tuple(x.shape[1:3]) else list(x.shape[1:3])
        metadata["resize"] = tuple(resize) if resize else None

        iscell = cellprob_i > cellprob_threshold
        inds = np.nonzero(iscell)
        metadata["cell_pixels"] = int(len(inds[0]))
        if len(inds[0]) == 0:
            masks = np.zeros(cellprob_i.shape, dtype=np.uint16)
        else:
            p_final = timed(
                timings,
                "follow_flows",
                device,
                lambda: dynamics.follow_flows(
                    dP_i * iscell / 5.0,
                    inds=inds,
                    niter=niter_resolved,
                    interp=interp,
                    device=device,
                ),
            )
            if not torch.is_tensor(p_final):
                p_final = torch.from_numpy(p_final).to(device, dtype=torch.int)
            else:
                p_final = p_final.int()
            if device.type == "mps":
                p_final = p_final.to(torch.device("cpu"))

            masks = timed(
                timings,
                "get_masks_torch",
                device,
                lambda: dynamics.get_masks_torch(
                    p_final,
                    inds,
                    dP_i.shape[1:],
                    max_size_fraction=0.4,
                ),
            )
            del p_final

            if masks.max() > 0 and flow_threshold is not None and flow_threshold > 0:
                masks = timed(
                    timings,
                    "remove_bad_flow_masks",
                    device,
                    lambda: dynamics.remove_bad_flow_masks(
                        masks,
                        dP_i,
                        threshold=flow_threshold,
                        device=device,
                    ),
                )
            else:
                timings["remove_bad_flow_masks"] = 0.0

        if resize is not None:
            masks = timed(
                timings,
                "resize_masks",
                device,
                lambda: transforms.resize_image(
                    masks,
                    resize[0],
                    resize[1],
                    no_channels=True,
                    interpolation=cv2.INTER_NEAREST,
                ),
            )
        else:
            timings["resize_masks"] = 0.0

        masks = timed(
            timings,
            "fill_holes_remove_small",
            device,
            lambda: utils.fill_holes_and_remove_small_masks(masks, min_size=15),
        )
        timed(timings, "flow_display", device, lambda: plot.dx_to_circ(dP.squeeze()))

        synchronize(device)
        timings["total"] = time.perf_counter() - total_start
        metadata["labels"] = int(masks.max(initial=0))
        metadata["mask_shape"] = tuple(int(v) for v in masks.shape)
        metadata["mask_dtype"] = str(masks.dtype)

    return ProfileResult(name=name, device=str(device), timings=timings, metadata=metadata, masks=masks)


def profile_stardist(image: np.ndarray, device_name: str) -> ProfileResult:
    from cistardist_pytorch import StarDist2D
    from cistardist_pytorch.geometry import dist_to_coord, polygons_to_label
    from cistardist_pytorch.model import _crop_prediction
    from cistardist_pytorch.nms import _candidate_mask, non_maximum_suppression

    model = StarDist2D.from_folder(str(SD_MODEL_DIR), device=device_name)
    device = model.device
    timings: dict[str, float] = {}
    metadata: dict[str, Any] = {
        "image_shape": tuple(int(v) for v in image.shape),
        "image_dtype": str(image.dtype),
    }
    total_start = time.perf_counter()

    x, original_shape, pad = timed(
        timings,
        "prepare_normalize_pad",
        device,
        lambda: model._prepare_image(image, normalize=True),
    )
    metadata["padded_shape"] = tuple(int(v) for v in x.shape)
    metadata["pad"] = tuple(int(v) for v in pad)

    tensor = timed(
        timings,
        "numpy_to_tensor_device",
        device,
        lambda: torch.from_numpy(x[None, None]).to(device),
    )

    def network_forward() -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return model.net(tensor)

    prob_t, dist_t = timed(
        timings,
        "network_forward",
        device,
        network_forward,
    )

    def outputs_to_numpy() -> tuple[np.ndarray, np.ndarray]:
        prob = prob_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        dist = np.moveaxis(dist_t[0].detach().cpu().numpy(), 0, -1).astype(np.float32, copy=False)
        return (
            _crop_prediction(prob, pad, model.config.grid),
            _crop_prediction(dist, pad, model.config.grid),
        )

    prob, dist = timed(timings, "output_to_cpu_crop", device, outputs_to_numpy)
    dist = timed(timings, "distance_clip", device, lambda: np.maximum(dist, 1e-3))

    prob_thresh = model.thresholds["prob"]
    nms_thresh = model.thresholds["nms"]
    candidate_mask = timed(
        timings,
        "candidate_mask",
        device,
        lambda: _candidate_mask(prob, prob_thresh=prob_thresh, b=2),
    )
    metadata["candidate_pixels"] = int(np.count_nonzero(candidate_mask))

    points, probi, disti = timed(
        timings,
        "non_maximum_suppression",
        device,
        lambda: non_maximum_suppression(
            dist,
            prob,
            grid=model.config.grid,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
        ),
    )
    metadata["instances_after_nms"] = int(len(points))

    labels = timed(
        timings,
        "render_labels",
        device,
        lambda: polygons_to_label(disti, points, shape=original_shape, prob=probi),
    )
    timed(
        timings,
        "build_details_coord",
        device,
        lambda: dist_to_coord(disti, points)
        if len(points)
        else np.zeros((0, 2, model.config.n_rays), dtype=np.float32),
    )

    synchronize(device)
    timings["total"] = time.perf_counter() - total_start
    metadata["labels"] = int(labels.max(initial=0))
    metadata["mask_shape"] = tuple(int(v) for v in labels.shape)
    metadata["mask_dtype"] = str(labels.dtype)
    return ProfileResult("StarDist 2D", str(device), timings, metadata, masks=labels)


def object_metrics(ref: np.ndarray, pred: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    ref = ref.astype(np.int64, copy=False)
    pred = pred.astype(np.int64, copy=False)
    n_ref = int(ref.max()) + 1
    n_pred = int(pred.max()) + 1
    if n_ref <= 1 or n_pred <= 1:
        return {"matches": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0}

    overlap = np.bincount(
        ref.ravel() * n_pred + pred.ravel(),
        minlength=n_ref * n_pred,
    ).reshape(n_ref, n_pred)
    area_ref = overlap.sum(axis=1)
    area_pred = overlap.sum(axis=0)
    intersection = overlap[1:, 1:]
    union = area_ref[1:, None] + area_pred[None, 1:] - intersection
    iou = intersection / np.maximum(union, 1)

    candidates = []
    rows, cols = np.nonzero(iou >= threshold)
    for row, col in zip(rows, cols):
        candidates.append((float(iou[row, col]), int(row), int(col)))
    candidates.sort(reverse=True)

    used_ref, used_pred, matched_iou = set(), set(), []
    for value, row, col in candidates:
        if row not in used_ref and col not in used_pred:
            used_ref.add(row)
            used_pred.add(col)
            matched_iou.append(value)

    matches = len(matched_iou)
    ref_count = n_ref - 1
    pred_count = n_pred - 1
    precision = matches / pred_count if pred_count else 0.0
    recall = matches / ref_count if ref_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = float(np.mean(matched_iou)) if matched_iou else 0.0
    return {
        "matches": matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
    }


def print_result(result: ProfileResult) -> None:
    meta = result.metadata
    print()
    print(f"{result.name} | {result.device}")
    print(
        f"  image={meta['image_shape']} {meta['image_dtype']}  "
        f"mask={meta['mask_shape']} {meta['mask_dtype']}  labels={meta['labels']}"
    )
    for key in (
        "fast_mode",
        "diameter",
        "rescale",
        "resample",
        "interp",
        "niter",
        "flow_threshold",
        "cell_pixels",
        "candidate_pixels",
        "instances_after_nms",
    ):
        if key in meta:
            print(f"  {key}={meta[key]}")
    print("  step                         time")
    print("  ---------------------------  ------------")
    for name, seconds in sorted(result.timings.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name:<27}  {format_seconds(seconds)}")


def print_summary(results: list[ProfileResult]) -> None:
    print()
    print("Summary")
    print("-------")
    print(f"{'profile':<24}{'device':<10}{'labels':>8}  {'total':>12}")
    for result in results:
        print(
            f"{result.name:<24}{result.device:<10}"
            f"{result.metadata['labels']:>8}  {format_seconds(result.timings['total']):>12}"
        )

    print()
    print("CP3 Fast vs Default")
    print("-------------------")
    for default in results:
        if default.name != "CP3 default" or default.masks is None:
            continue
        for result in results:
            if (
                result.device != default.device
                or result is default
                or result.masks is None
                or not result.name.startswith("CP3 ")
            ):
                continue
            metrics = object_metrics(default.masks, result.masks)
            print(
                f"{result.name:<24}{result.device:<10}"
                f"count={result.metadata['labels']:>4}/{default.metadata['labels']:<4}  "
                f"F1={metrics['f1']:.4f}  "
                f"P={metrics['precision']:.4f}  "
                f"R={metrics['recall']:.4f}  "
                f"mIoU={metrics['mean_iou']:.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile CP3 and StarDist segmentation stages.")
    parser.add_argument("--image", type=Path, default=IMAGE_PATH)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument("--gpu_only", action="store_true")
    parser.add_argument("--skip_stardist", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu_only:
        args.devices = [d for d in args.devices if d.startswith("cuda")]
    if not args.image.exists():
        raise FileNotFoundError(args.image)

    image = tifffile.imread(args.image)
    results: list[ProfileResult] = []
    for device_name in args.devices:
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            print(f"Skipping {device_name}: CUDA is not available")
            continue

        results.append(profile_cp3(image, device_name, "CP3 default"))
        results.append(profile_cp3(image, device_name, "CP3 fast n50", fast_mode=True, niter=50))
        results.append(profile_cp3(image, device_name, "CP3 fast n100", fast_mode=True, niter=100))
        if not args.skip_stardist:
            results.append(profile_stardist(image, device_name))

    for result in results:
        print_result(result)
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
