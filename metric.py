import os
import math
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from scipy import stats


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * b[k].astype(int) + a[k], minlength=n ** 2).reshape(n, n)

def cal_kappa(hist):
    if hist.sum() == 0:
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


DEFAULT_CLASS_NAMES: List[str] = [
    "building",
    "water",
    "low_vegetation",
    "tree",
    "ground",
    "playground",
]
DEFAULT_CLASS_COLORS: List[Tuple[int, int, int]] = [
    (128, 0, 0),
    (0, 0, 255),
    (0, 128, 0),
    (0, 255, 0),
    (128, 128, 128),
    (255, 0, 0),
]


def _to_rgb_array(img_like: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """Convert input (path/np array/PIL) to RGB uint8 array of shape (H, W, 3).

    Accepts:
      - str: file path to an image
      - np.ndarray: either HxW (grayscale), HxWx3/4; assumed uint8-like
      - PIL.Image.Image
    """
    if isinstance(img_like, str):
        if not os.path.exists(img_like):
            raise FileNotFoundError(f"Image path not found: {img_like}")
        pil_img = Image.open(img_like).convert("RGB")
        return np.asarray(pil_img, dtype=np.uint8)

    if isinstance(img_like, Image.Image):
        return np.asarray(img_like.convert("RGB"), dtype=np.uint8)

    if isinstance(img_like, np.ndarray):
        arr = img_like
        if arr.ndim == 2:
            # grayscale -> stack to RGB
            return np.stack([arr, arr, arr], axis=-1).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            # if RGBA, drop alpha
            if arr.shape[2] == 4:
                arr = arr[..., :3]
            # assume already RGB; if BGR is possible upstream, caller should convert
            return arr.astype(np.uint8)
        raise ValueError(f"Unsupported ndarray shape: {arr.shape}")

    raise TypeError(f"Unsupported input type: {type(img_like)}")


def make_color_to_index(
    class_colors: Sequence[Tuple[int, int, int]] = DEFAULT_CLASS_COLORS,
) -> Dict[Tuple[int, int, int], int]:
    """Build mapping from RGB color to class index (1..N). Index 0 is reserved."""
    color_to_idx: Dict[Tuple[int, int, int], int] = {}
    for idx, rgb in enumerate(class_colors, start=1):
        color_to_idx[tuple(int(v) for v in rgb)] = idx
    return color_to_idx


def color_gt_to_index_map(
    color_img: Union[str, np.ndarray, Image.Image],
    color_to_index: Dict[Tuple[int, int, int], int],
) -> np.ndarray:
    """Convert a color semantic GT image to an index map with classes 1..N.

    Pixels whose color does not match any known class are set to 0.
    """
    rgb = _to_rgb_array(color_img)
    h, w, _ = rgb.shape
    index_map = np.zeros((h, w), dtype=np.int64)

    # vectorized assignment per color
    for color_rgb, cls_idx in color_to_index.items():
        mask = (
            (rgb[..., 0] == color_rgb[0])
            & (rgb[..., 1] == color_rgb[1])
            & (rgb[..., 2] == color_rgb[2])
        )
        if mask.any():
            index_map[mask] = cls_idx
    return index_map


def apply_change_mask(index_map: np.ndarray, change_mask: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """Zero-out pixels where change_mask==0. Keeps class indices elsewhere.

    change_mask is expected to be binary-like (0/1). Any non-zero is treated as 1.
    """
    cm = _to_rgb_array(change_mask)
    # if change mask is 3-channel, reduce to single channel
    if cm.ndim == 3:
        # take any channel since it's binary; turn to single-channel boolean
        cm = cm[..., 0]
    if cm.ndim != 2:
        raise ValueError(f"change_mask must be HxW (or convertible), got shape {cm.shape}")

    if cm.shape != index_map.shape:
        raise ValueError(
            f"Shape mismatch between index map {index_map.shape} and change mask {cm.shape}"
        )

    out = index_map.copy()
    keep = cm.astype(np.uint8) > 0
    out[~keep] = 0
    return out


def preprocess_semantic_gt(
    label_color_img: Union[str, np.ndarray, Image.Image],
    gt_change_mask: Union[str, np.ndarray, Image.Image],
    class_colors: Sequence[Tuple[int, int, int]] = DEFAULT_CLASS_COLORS,
) -> np.ndarray:
    """Convert a color GT into an index map (1..N) and zero-out unchanged (0).

    Returns an array of shape (H, W), dtype int64, with values in {0..N}.
    """
    color_to_index = make_color_to_index(class_colors)
    idx = color_gt_to_index_map(label_color_img, color_to_index)
    idx_masked = apply_change_mask(idx, gt_change_mask)
    return idx_masked


def preprocess_semantic_gt_list(
    label_list: Sequence[Union[str, np.ndarray, Image.Image]],
    change_mask_list: Sequence[Union[str, np.ndarray, Image.Image]],
    class_colors: Sequence[Tuple[int, int, int]] = DEFAULT_CLASS_COLORS,
) -> List[np.ndarray]:
    """Batch version for multiple labels and change masks (aligned by index)."""
    if len(label_list) != len(change_mask_list):
        raise ValueError(
            f"label_list and change_mask_list must have same length, got {len(label_list)} vs {len(change_mask_list)}"
        )
    color_to_index = make_color_to_index(class_colors)
    outputs: List[np.ndarray] = []
    for lbl, cm in zip(label_list, change_mask_list):
        idx = color_gt_to_index_map(lbl, color_to_index)
        idx = apply_change_mask(idx, cm)
        outputs.append(idx)
    return outputs


def preprocess_gt_pair(
    label1_list: Sequence[Union[str, np.ndarray, Image.Image]],
    label2_list: Sequence[Union[str, np.ndarray, Image.Image]],
    gt_masks: Sequence[Union[str, np.ndarray, Image.Image]],
    class_colors: Sequence[Tuple[int, int, int]] = DEFAULT_CLASS_COLORS,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Preprocess two lists of color GTs (time1/time2) with the same gt change masks.

    - label1_list[i] and label2_list[i] are color semantic GTs for sample i
    - gt_masks[i] is the binary change GT; 0=unchanged, 1=changed
    Returns (label1_indices, label2_indices), each a list of HxW int64 arrays
    with values in {0..N} where 0 marks unchanged pixels.
    """
    if not (len(label1_list) == len(label2_list) == len(gt_masks)):
        raise ValueError(
            "label1_list, label2_list, and gt_masks must have the same length"
        )
    color_to_index = make_color_to_index(class_colors)

    idx_list_1: List[np.ndarray] = []
    idx_list_2: List[np.ndarray] = []
    for lbl1, lbl2, cm in zip(label1_list, label2_list, gt_masks):
        idx1 = color_gt_to_index_map(lbl1, color_to_index)
        idx2 = color_gt_to_index_map(lbl2, color_to_index)
        idx1 = apply_change_mask(idx1, cm)
        idx2 = apply_change_mask(idx2, cm)
        idx_list_1.append(idx1)
        idx_list_2.append(idx2)

    return idx_list_1, idx_list_2


# Convenience constant for number of classes including 0 (unchanged)
NUM_SCD_CLASSES: int = 1 + len(DEFAULT_CLASS_COLORS)




def _pred_to_indices_single(pred_masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a single prediction tensor (13,H,W) into indices and change mask.

    Layout:
      - pred_masks[0] is the binary-like change map (0=unchanged, 1=changed). Any >0 treated as 1.
      - pred_masks[1:7] are 6 channels for time-1 classes in the order of DEFAULT_CLASS_NAMES
      - pred_masks[7:13] are 6 channels for time-2 classes in the same order

    Returns (idx_t1, idx_t2, change_mask):
      - idx_t1, idx_t2: HxW int64 with values in {0..6}, where 0 marks unchanged
      - change_mask: HxW uint8 in {0,1}
    """
    if not isinstance(pred_masks, np.ndarray):
        raise TypeError(f"pred_masks must be np.ndarray, got {type(pred_masks)}")
    if pred_masks.ndim != 3:
        raise ValueError(f"pred_masks must have shape (13,H,W), got ndim={pred_masks.ndim}")
    if pred_masks.shape[0] != 13:
        raise ValueError(f"pred_masks must have 13 channels, got {pred_masks.shape[0]}")

    _, h, w = pred_masks.shape

    # change mask
    change_ch = pred_masks[0]
    change_mask = (change_ch > 0).astype(np.uint8)

    # time-1 and time-2 logits/scores
    t1 = pred_masks[1:7]   # shape (6,H,W)
    t2 = pred_masks[7:13]  # shape (6,H,W)

    # 对每个像素位置的类别维度做softmax归一化
    def softmax(x, axis=0):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    t1_probs = softmax(t1, axis=0)  # shape (6,H,W)
    t2_probs = softmax(t2, axis=0)  # shape (6,H,W)
    t1_argmax = np.argmax(t1_probs, axis=0).astype(np.int64) + 1
    t2_argmax = np.argmax(t2_probs, axis=0).astype(np.int64) + 1

    # apply change mask: where change==0, set to 0
    idx_t1 = t1_argmax.copy()
    idx_t2 = t2_argmax.copy()
    idx_t1[change_mask == 0] = 0
    idx_t2[change_mask == 0] = 0

    return idx_t1, idx_t2, change_mask


def preds_to_indices_batch(
    pred_masks_list: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Batch process a list of prediction tensors.

    Each element is a (13,H,W) array as described in _pred_to_indices_single.
    Returns (idx_list_t1, idx_list_t2, change_masks).
    """
    idx_list_t1: List[np.ndarray] = []
    idx_list_t2: List[np.ndarray] = []
    change_masks: List[np.ndarray] = []
    for pm in pred_masks_list:
        idx_t1, idx_t2, cm = _pred_to_indices_single(pm)
        idx_list_t1.append(idx_t1)
        idx_list_t2.append(idx_t2)
        change_masks.append(cm)
    return idx_list_t1, idx_list_t2, change_masks


# =========================
# Confusion & SCD Metrics
# =========================

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    mask: np.ndarray = None,
    ignore_zero: bool = True,
) -> np.ndarray:
    """Compute KxK confusion matrix for indices in {0..K} or {1..K}.

    - y_true, y_pred: HxW or flat arrays of same shape
    - num_classes: K (excluding 0)
    - mask: optional boolean mask; if provided, only masked==True pixels are used
    - ignore_zero: if True, pixels where either y_true==0 or y_pred==0 are ignored
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    t = y_true.reshape(-1)
    p = y_pred.reshape(-1)
    m = np.ones_like(t, dtype=bool)
    if mask is not None:
        m &= mask.reshape(-1).astype(bool)
    if ignore_zero:
        m &= (t > 0) & (p > 0)

    t = t[m]
    p = p[m]
    if t.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    # Shift to 0-based
    t0 = t - 1 if ignore_zero else t
    p0 = p - 1 if ignore_zero else p

    cm = np.bincount(t0 * (num_classes) + p0, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def overall_accuracy(cm: np.ndarray) -> float:
    """OA = sum(diagonal) / total."""
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.trace(cm)) / float(total)


def kappa_score(cm: np.ndarray) -> float:
    """Cohen's Kappa computed from confusion matrix."""
    total = cm.sum()
    if total == 0:
        return 0.0
    po = np.trace(cm) / total
    row_marginals = cm.sum(axis=1)
    col_marginals = cm.sum(axis=0)
    pe = float(np.dot(row_marginals, col_marginals)) / float(total * total)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def f1_change_from_indices(
    gt_t1: np.ndarray,
    gt_t2: np.ndarray,
    pred_t1: np.ndarray,
    pred_t2: np.ndarray,
    valid_mask: np.ndarray = None,
) -> float:
    """Compute F1 for change detection derived from indices.

    - Change is defined as (label at t1 != label at t2)
    - Pixels with any label==0 are ignored
    - Optionally further restricted by valid_mask
    """
    if not (gt_t1.shape == gt_t2.shape == pred_t1.shape == pred_t2.shape):
        raise ValueError("All inputs must have the same shape")

    h, w = gt_t1.shape
    gt_change = (gt_t1 != gt_t2) & (gt_t1 > 0) & (gt_t2 > 0)
    pred_change = (pred_t1 != pred_t2) & (pred_t1 > 0) & (pred_t2 > 0)

    mask = np.ones((h, w), dtype=bool)
    if valid_mask is not None:
        mask &= valid_mask.astype(bool)

    # Flatten
    g = gt_change[mask].reshape(-1)
    p = pred_change[mask].reshape(-1)
    if g.size == 0:
        return 0.0

    tp = np.logical_and(g, p).sum()
    fp = np.logical_and(~g, p).sum()
    fn = np.logical_and(g, ~p).sum()

    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return float(2 * tp) / float(denom)


def compute_scd_metrics(
    gt_t1: np.ndarray,
    gt_t2: np.ndarray,
    pred_t1: np.ndarray,
    pred_t2: np.ndarray,
    num_classes: int = 6,
    which_of_time: str = "t2",
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute SCD metrics: OA, SeK (kappa), Fscd.

    - Confusion is computed on semantic indices of selected time (t1 or t2)
    - Only changed pixels are considered for semantic metrics (OA/SeK)
    - Pixels with label==0 are ignored
    - Fscd is F1 on change detection derived from indices
    """
    if which_of_time not in ("t1", "t2"):
        raise ValueError("which_of_time must be 't1' or 't2'")
    if not (gt_t1.shape == gt_t2.shape == pred_t1.shape == pred_t2.shape):
        raise ValueError("All inputs must have the same shape")

    # Changed-pixel mask based on GT
    change_mask = (gt_t1 != gt_t2) & (gt_t1 > 0) & (gt_t2 > 0)

    gt_sel = gt_t1 if which_of_time == "t1" else gt_t2
    pred_sel = pred_t1 if which_of_time == "t1" else pred_t2

    cm = compute_confusion_matrix(
        gt_sel,
        pred_sel,
        num_classes=num_classes,
        mask=change_mask,
        ignore_zero=True,
    )
    oa = overall_accuracy(cm)
    
    cm_for_sek = cm.copy()
    cm_for_sek[0, 0] = 0 
    sek = kappa_score(cm_for_sek)
    
    fscd = f1_change_from_indices(gt_t1, gt_t2, pred_t1, pred_t2)

    return {
        "confusion": cm,
        "OA": oa,
        "SeK": sek,
        "Fscd": fscd,
    }


def compute_scd_metrics_gstm_style(
    gt_t1: np.ndarray,
    gt_t2: np.ndarray,
    pred_t1: np.ndarray,
    pred_t2: np.ndarray,
    num_classes: int = 7,
) -> np.ndarray:
    if not (gt_t1.shape == gt_t2.shape == pred_t1.shape == pred_t2.shape):
        raise ValueError("All inputs must have the same shape")

    preds_all = [pred_t1.flatten(), pred_t2.flatten()]
    labels_all = [gt_t1.flatten(), gt_t2.flatten()]
    
    hist = np.zeros((num_classes, num_classes))
    for pred, label in zip(preds_all, labels_all):
        infer_array = np.array(pred)
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist += fast_hist(infer_array, label_array, num_classes)
    
    return hist


