from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tif
import torch
from scipy import ndimage as ndi

from .models import UNet2D, UNet3D


@dataclass(frozen=True)
class ModelConfig:
    ndim: int
    checkpoint: str
    channels: int = 16
    encoder_norm: str | None = "instance"
    decoder_norm: str | None = None
    vf_clip: float = 14.0
    discard_value: float = 3.0
    threshold: float = 5.0
    min_size: int = 21
    threshold_quantize_to_uint8: bool = False
    label_connectivity: int | None = None
    kernel_size: tuple[int, int, int] = (3, 3, 3)
    padding: tuple[int, int, int] = (1, 1, 1)
    use_separable_conv: bool = False
    assignment_radius: float | None = None
    normalization: str = "minmax_quantile"
    segmentation_backend: str = "compact"


def load_tif(path: str | Path) -> np.ndarray:
    arr = tif.imread(str(path)).astype(np.float32)
    return arr


def summarize_input(arr: np.ndarray, ndim: int) -> np.ndarray:
    if ndim == 2 and arr.ndim == 3:
        return arr.mean(axis=0).astype(np.float32)
    if ndim == 3 and arr.ndim == 4:
        return arr.mean(axis=0).astype(np.float32)
    return arr.astype(np.float32, copy=False)


def normalize_image(arr: np.ndarray, method: str = "minmax_quantile") -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    if method in ("identity", "none", None):
        return arr
    if method == "quantile_01_99":
        q01, q99 = np.quantile(arr, [0.01, 0.99])
        denom = max(float(q99 - q01), 1e-6)
        return ((arr - q01) / denom).astype(np.float32)
    if method != "minmax_quantile":
        raise ValueError(f"Unsupported normalization method: {method}")
    arr = arr - np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx > 0:
        arr = arr / mx
    q01, q99 = np.quantile(arr, [0.01, 0.99])
    denom = max(float(q99 - q01), 1e-6)
    arr = (arr - q01) / denom
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def build_model(config: ModelConfig, device: torch.device) -> torch.nn.Module:
    features = tuple(config.channels * (2**i) for i in range(4))
    if config.ndim == 2:
        model = UNet2D(
            in_channels=1,
            out_channels=2,
            features=features,
            encoder_norm=config.encoder_norm,
            decoder_norm=config.decoder_norm,
            residual=True,
        )
    elif config.ndim == 3:
        model = UNet3D(
            in_channels=1,
            out_channels=3,
            features=features,
            encoder_norm=config.encoder_norm,
            decoder_norm=config.decoder_norm,
            residual=True,
            kernel_size=tuple(config.kernel_size),
            padding=tuple(config.padding),
            use_separable=config.use_separable_conv,
        )
    else:
        raise ValueError("ndim must be 2 or 3.")

    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


@torch.no_grad()
def predict_vector_field(image: np.ndarray, config: ModelConfig, device: str = "auto") -> np.ndarray:
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model = build_model(config, dev)
    arr = normalize_image(image, config.normalization)
    if config.ndim == 2:
        tensor = torch.from_numpy(arr)[None, None].to(dev)
    else:
        tensor = torch.from_numpy(arr)[None, None].to(dev)

    vf = model(tensor).clamp(-config.vf_clip, config.vf_clip)
    return vf[0].detach().cpu().numpy().astype(np.float32)


def _move_values_2d(board: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    h, w = board.shape
    out = torch.zeros_like(board)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    yy, xx = yy.to(board.device), xx.to(board.device)
    new_y = yy.float() + field[..., 1]
    new_x = xx.float() + field[..., 0]
    low_y, high_y = torch.floor(new_y).long(), torch.ceil(new_y).long()
    low_x, high_x = torch.floor(new_x).long(), torch.ceil(new_x).long()
    wy_hi, wx_hi = new_y - low_y.float(), new_x - low_x.float()
    wy_lo, wx_lo = 1 - wy_hi, 1 - wx_hi
    low_y, high_y = low_y.clamp(0, h - 1), high_y.clamp(0, h - 1)
    low_x, high_x = low_x.clamp(0, w - 1), high_x.clamp(0, w - 1)
    flat_w = w
    flat = out.flatten()
    flat.scatter_add_(0, (low_y * flat_w + low_x).flatten(), (board * wy_lo * wx_lo).flatten())
    flat.scatter_add_(0, (high_y * flat_w + low_x).flatten(), (board * wy_hi * wx_lo).flatten())
    flat.scatter_add_(0, (low_y * flat_w + high_x).flatten(), (board * wy_lo * wx_hi).flatten())
    flat.scatter_add_(0, (high_y * flat_w + high_x).flatten(), (board * wy_hi * wx_hi).flatten())
    return flat.reshape(h, w)


def _move_values_3d(board: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    d, h, w = board.shape
    out = torch.zeros_like(board)
    zz, yy, xx = torch.meshgrid(torch.arange(d), torch.arange(h), torch.arange(w), indexing="ij")
    zz, yy, xx = zz.to(board.device), yy.to(board.device), xx.to(board.device)
    nz = zz.float() + field[..., 2]
    ny = yy.float() + field[..., 1]
    nx = xx.float() + field[..., 0]
    lz, hz = torch.floor(nz).long(), torch.ceil(nz).long()
    ly, hy = torch.floor(ny).long(), torch.ceil(ny).long()
    lx, hx = torch.floor(nx).long(), torch.ceil(nx).long()
    wz_h, wy_h, wx_h = nz - lz.float(), ny - ly.float(), nx - lx.float()
    wz_l, wy_l, wx_l = 1 - wz_h, 1 - wy_h, 1 - wx_h
    lz, hz = lz.clamp(0, d - 1), hz.clamp(0, d - 1)
    ly, hy = ly.clamp(0, h - 1), hy.clamp(0, h - 1)
    lx, hx = lx.clamp(0, w - 1), hx.clamp(0, w - 1)

    def add(z, y, x, weight):
        out.flatten().scatter_add_(0, (z * h * w + y * w + x).flatten(), (board * weight).flatten())

    add(lz, ly, lx, wz_l * wy_l * wx_l)
    add(hz, ly, lx, wz_h * wy_l * wx_l)
    add(lz, hy, lx, wz_l * wy_h * wx_l)
    add(lz, ly, hx, wz_l * wy_l * wx_h)
    add(hz, hy, lx, wz_h * wy_h * wx_l)
    add(hz, ly, hx, wz_h * wy_l * wx_h)
    add(lz, hy, hx, wz_l * wy_h * wx_h)
    add(hz, hy, hx, wz_h * wy_h * wx_h)
    return out


def vector_field_to_mask(vf: np.ndarray, image_shape: tuple[int, ...], config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.segmentation_backend == "legacy_train2d":
        if config.ndim != 2:
            raise ValueError("legacy_train2d segmentation is only defined for 2D models.")
        return _legacy_train2d_vector_field_to_mask(vf, image_shape, config)

    structure = None
    if config.label_connectivity is not None:
        structure = ndi.generate_binary_structure(config.ndim, int(config.label_connectivity))

    if config.ndim == 2:
        field = torch.from_numpy(np.moveaxis(vf, 0, -1))
        board = torch.ones(image_shape, dtype=torch.float32)
        board_reg = _move_values_2d(board, field)
        board_np = _threshold_board(board_reg, config)
        labels, _ = ndi.label(board_np > 0, structure=structure)
        yy, xx = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]), indexing="ij")
        moved_y = yy + field[..., 1].numpy()
        moved_x = xx + field[..., 0].numpy()
        y_new = np.rint(moved_y).clip(0, image_shape[0] - 1).astype(np.int32)
        x_new = np.rint(moved_x).clip(0, image_shape[1] - 1).astype(np.int32)
        mask = labels[y_new, x_new].astype(np.int32)
        if config.assignment_radius is not None:
            dist2 = (moved_y - y_new.astype(np.float32)) ** 2 + (moved_x - x_new.astype(np.float32)) ** 2
            mask[dist2 > float(config.assignment_radius) ** 2] = 0
    else:
        field = torch.from_numpy(np.moveaxis(vf, 0, -1))
        board = torch.ones(image_shape, dtype=torch.float32)
        board_reg = _move_values_3d(board, field)
        board_np = _threshold_board(board_reg, config)
        labels, _ = ndi.label(board_np > 0, structure=structure)
        zz, yy, xx = np.meshgrid(
            np.arange(image_shape[0]), np.arange(image_shape[1]), np.arange(image_shape[2]), indexing="ij"
        )
        moved_z = zz + field[..., 2].numpy()
        moved_y = yy + field[..., 1].numpy()
        moved_x = xx + field[..., 0].numpy()
        z_new = np.rint(moved_z).clip(0, image_shape[0] - 1).astype(np.int32)
        y_new = np.rint(moved_y).clip(0, image_shape[1] - 1).astype(np.int32)
        x_new = np.rint(moved_x).clip(0, image_shape[2] - 1).astype(np.int32)
        mask = labels[z_new, y_new, x_new].astype(np.int32)
        if config.assignment_radius is not None:
            dist2 = (
                (moved_z - z_new.astype(np.float32)) ** 2
                + (moved_y - y_new.astype(np.float32)) ** 2
                + (moved_x - x_new.astype(np.float32)) ** 2
            )
            mask[dist2 > float(config.assignment_radius) ** 2] = 0

    mask = filter_and_reindex(mask, min_size=config.min_size)
    return mask, board_np


def _legacy_train2d_vector_field_to_mask(
    vf: np.ndarray, image_shape: tuple[int, ...], config: ModelConfig
) -> tuple[np.ndarray, np.ndarray]:
    field = torch.from_numpy(np.moveaxis(vf, 0, -1).astype(np.float32, copy=False))
    board = torch.ones(image_shape, dtype=torch.float32)
    board_reg = _move_values_2d(board, field)
    board_reg = (board_reg - config.discard_value).clamp(min=0)
    mx = board_reg.max()
    if mx > 0:
        board_reg = board_reg / mx * 255
    board_u8 = board_reg.numpy().astype(np.uint8)
    board_thr = (board_u8 > config.threshold).astype(np.uint8)

    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    connected, n_labels = ndi.label(board_thr > 0, structure=structure)

    yy, xx = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
    moved_x = yy.astype(np.float32) + field[..., 0].numpy()
    moved_y = xx.astype(np.float32) + field[..., 1].numpy()
    radius = 1.0 if config.assignment_radius is None else float(config.assignment_radius)
    radius_sq = radius * radius

    final_masks = np.zeros(image_shape, dtype=np.int32)
    for label_id in range(1, int(n_labels) + 1):
        coords_y, coords_x = np.nonzero(connected == label_id)
        if coords_y.size == 0:
            continue
        component_mask = np.zeros(image_shape, dtype=bool)
        for cy, cx in zip(coords_y, coords_x):
            dy = moved_y - float(cy)
            dx = moved_x - float(cx)
            component_mask |= (dx * dx + dy * dy) < radius_sq
        final_masks = np.maximum(final_masks, component_mask.astype(np.int32) * label_id)

    return final_masks, board_thr


def _threshold_board(board: torch.Tensor, config: ModelConfig) -> np.ndarray:
    board = (board - config.discard_value).clamp(min=0)
    mx = board.max()
    if mx > 0:
        board = board / mx * 255
    board_np = board.cpu().numpy()
    if config.threshold_quantize_to_uint8:
        board_np = board_np.astype(np.uint8)
    return (board_np > config.threshold).astype(np.uint8)


def filter_and_reindex(mask: np.ndarray, min_size: int = 21) -> np.ndarray:
    mask = mask.astype(np.int64, copy=False)
    counts = np.bincount(mask.ravel())
    keep = counts >= int(min_size)
    if keep.size:
        keep[0] = False
    kept_ids = np.flatnonzero(keep)
    lut = np.zeros_like(counts, dtype=np.int32)
    lut[kept_ids] = np.arange(1, len(kept_ids) + 1, dtype=np.int32)
    return lut[mask]


def infer_tif(
    input_path: str | Path,
    output_dir: str | Path,
    config: ModelConfig,
    device: str = "auto",
    save_vector_field: bool = False,
) -> dict[str, Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = summarize_input(load_tif(input_path), config.ndim)
    vf = predict_vector_field(image, config, device=device)
    mask, board = vector_field_to_mask(vf, image.shape, config)

    stem = input_path.stem
    paths = {
        "mask": output_dir / f"{stem}_profile_mask.tif",
        "board": output_dir / f"{stem}_profile_board.tif",
    }
    tif.imwrite(paths["mask"], mask.astype(np.uint16 if mask.max() < 65535 else np.uint32))
    tif.imwrite(paths["board"], board.astype(np.uint8))
    if save_vector_field:
        paths["vector_field"] = output_dir / f"{stem}_profile_vector_field.tif"
        tif.imwrite(paths["vector_field"], vf.astype(np.float32))
    return paths


def f1_vs_iou(gt: np.ndarray, pred: np.ndarray, thresholds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if thresholds is None:
        thresholds = np.linspace(0, 1, 11)
    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)
    gt_ids = np.array([x for x in np.unique(gt) if x != 0], dtype=np.int64)
    pred_ids = np.array([x for x in np.unique(pred) if x != 0], dtype=np.int64)
    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return thresholds, np.ones_like(thresholds, dtype=np.float32)
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return thresholds, np.zeros_like(thresholds, dtype=np.float32)

    gt_inverse = np.zeros(int(gt_ids.max()) + 1, dtype=np.int64)
    pred_inverse = np.zeros(int(pred_ids.max()) + 1, dtype=np.int64)
    gt_inverse[gt_ids] = np.arange(1, len(gt_ids) + 1)
    pred_inverse[pred_ids] = np.arange(1, len(pred_ids) + 1)
    gt_compact = gt_inverse[gt]
    pred_compact = pred_inverse[pred]

    gt_area = np.bincount(gt_compact.ravel(), minlength=len(gt_ids) + 1).astype(np.float64)
    pred_area = np.bincount(pred_compact.ravel(), minlength=len(pred_ids) + 1).astype(np.float64)

    both = (gt_compact > 0) & (pred_compact > 0)
    pair_index = gt_compact[both].astype(np.int64) * (len(pred_ids) + 1) + pred_compact[both].astype(np.int64)
    unique_pairs, intersections = np.unique(pair_index, return_counts=True)

    ious = []
    for pair, inter in zip(unique_pairs, intersections):
        gi = int(pair // (len(pred_ids) + 1))
        pi = int(pair % (len(pred_ids) + 1))
        union = gt_area[gi] + pred_area[pi] - inter
        if union > 0:
            ious.append((inter / union, gi, pi))

    scores = []
    for thr in thresholds:
        used_g, used_p, tp = set(), set(), 0
        for iou, gi, pi in sorted(ious, reverse=True):
            if iou < thr or gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            tp += 1
        fp = len(pred_ids) - tp
        fn = len(gt_ids) - tp
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return thresholds, np.asarray(scores, dtype=np.float32)
