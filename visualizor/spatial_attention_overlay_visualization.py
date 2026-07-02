import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def _normalize_map(x, eps=1e-8, percentile=99.5):
    """
    Normalize attention map to [0, 1] using percentile clipping.
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    lo = np.percentile(x, 0.0)
    hi = np.percentile(x, percentile)

    if hi <= lo + eps:
        hi = x.max()

    if hi <= lo + eps:
        return np.zeros_like(x, dtype=np.float32)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + eps)
    return x


def _best_factor_pair(n, target_aspect):
    """
    Find h, w such that h * w == n and w / h is close to target_aspect.
    Used to approximately reshape image-token sequence into 2D grid.
    """
    best_h, best_w = 1, n
    best_err = float("inf")

    for h in range(1, int(math.sqrt(n)) + 1):
        if n % h == 0:
            w = n // h
            aspect = w / h
            err = abs(math.log((aspect + 1e-8) / (target_aspect + 1e-8)))

            if err < best_err:
                best_err = err
                best_h, best_w = h, w

    return best_h, best_w


def image_token_attention_to_2d(
    image_token_attn,
    image,
    force_grid=None,
    percentile=99.5,
):
    """
    Convert 1D image-token attention into an approximate 2D attention map.

    image_token_attn:
        shape [num_image_tokens]

    image:
        PIL.Image

    force_grid:
        optional tuple (grid_h, grid_w). If None, infer factor pair
        matching original image aspect ratio.

    return:
        heatmap_2d: shape [grid_h, grid_w], normalized to [0, 1]
    """
    attn = np.asarray(image_token_attn, dtype=np.float32)
    attn = np.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

    w, h = image.size
    target_aspect = w / h

    n = len(attn)

    if force_grid is not None:
        grid_h, grid_w = force_grid
        target_n = grid_h * grid_w

        if n >= target_n:
            attn = attn[:target_n]
        else:
            pad = np.zeros(target_n - n, dtype=np.float32)
            attn = np.concatenate([attn, pad], axis=0)

    else:
        grid_h, grid_w = _best_factor_pair(n, target_aspect)

    heatmap_2d = attn.reshape(grid_h, grid_w)
    heatmap_2d = _normalize_map(heatmap_2d, percentile=percentile)

    return heatmap_2d


def save_attention_overlay_on_image(
    image,
    heatmap_2d,
    save_path,
    title=None,
    alpha=0.45,
    cmap="jet",
):
    """
    Overlay 2D attention map on original image and save.

    image:
        PIL.Image

    heatmap_2d:
        2D numpy array, normalized or unnormalized

    save_path:
        output path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    image = image.convert("RGB")
    w, h = image.size

    heatmap_2d = _normalize_map(heatmap_2d)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.imshow(
        heatmap_2d,
        cmap=cmap,
        alpha=alpha,
        extent=(0, w, h, 0),
        interpolation="bilinear",
    )
    plt.axis("off")

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def get_image_token_attention_for_head(
    attention,
    input_range,
    layer_id,
    head_id,
):
    """
    Extract image-token attention for one layer-head pair.

    attention:
        attention_maps[0]
        list of layer attention tensors.
        each layer shape: [1, num_heads, query_len, key_len]

    input_range:
        (instruction_range, image_range)

    return:
        image_token_attn: shape [num_image_tokens]
    """
    _, image_range = input_range
    image_start, image_end = image_range

    attn_layer = attention[layer_id].to(torch.float32).cpu().numpy()

    image_token_attn = attn_layer[
        0,
        head_id,
        -1,
        image_start:image_end,
    ]

    image_token_attn = np.nan_to_num(
        image_token_attn,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return image_token_attn


def save_selected_head_spatial_overlay(
    clean_attention,
    attack_attention,
    clean_image,
    attack_image,
    clean_input_range,
    attack_input_range,
    selected_heads,
    save_dir="./selected_head_spatial_overlay",
    alpha=0.45,
    percentile=99.5,
):
    """
    Save spatial attention overlays for selected layer-head pairs.

    selected_heads:
        list of (layer_id, head_id, score) or (layer_id, head_id)

    Output:
        For each selected layer-head:
            clean_overlay.png
            attack_overlay.png
            attack_minus_clean_overlay.png  # approximate, after resizing maps
    """
    os.makedirs(save_dir, exist_ok=True)

    for item in selected_heads:
        if len(item) == 3:
            layer_id, head_id, score = item
        else:
            layer_id, head_id = item
            score = None

        head_dir = os.path.join(save_dir, f"L{layer_id:02d}_H{head_id:02d}")
        os.makedirs(head_dir, exist_ok=True)

        clean_token_attn = get_image_token_attention_for_head(
            clean_attention,
            clean_input_range,
            layer_id,
            head_id,
        )

        attack_token_attn = get_image_token_attention_for_head(
            attack_attention,
            attack_input_range,
            layer_id,
            head_id,
        )

        clean_heatmap = image_token_attention_to_2d(
            clean_token_attn,
            clean_image,
            percentile=percentile,
        )

        attack_heatmap = image_token_attention_to_2d(
            attack_token_attn,
            attack_image,
            percentile=percentile,
        )

        title_score = "" if score is None else f" | score={score:.6f}"

        save_attention_overlay_on_image(
            clean_image,
            clean_heatmap,
            os.path.join(head_dir, "clean_overlay.png"),
            title=f"Clean L{layer_id} H{head_id}{title_score}",
            alpha=alpha,
        )

        save_attention_overlay_on_image(
            attack_image,
            attack_heatmap,
            os.path.join(head_dir, "attack_overlay.png"),
            title=f"Attack L{layer_id} H{head_id}{title_score}",
            alpha=alpha,
        )

        # approximate diff overlay:
        # resize both coarse maps to the same original attack image canvas through matplotlib extent.
        # Here we compute diff on normalized maps after resizing through PIL.
        attack_w, attack_h = attack_image.size

        clean_map_img = Image.fromarray((clean_heatmap * 255).astype(np.uint8)).resize(
            (attack_w, attack_h),
            resample=Image.BILINEAR,
        )
        attack_map_img = Image.fromarray((attack_heatmap * 255).astype(np.uint8)).resize(
            (attack_w, attack_h),
            resample=Image.BILINEAR,
        )

        clean_map_resized = np.asarray(clean_map_img).astype(np.float32) / 255.0
        attack_map_resized = np.asarray(attack_map_img).astype(np.float32) / 255.0

        diff_map = attack_map_resized - clean_map_resized
        diff_map = np.maximum(diff_map, 0.0)
        diff_map = _normalize_map(diff_map, percentile=percentile)

        save_attention_overlay_on_image(
            attack_image,
            diff_map,
            os.path.join(head_dir, "attack_minus_clean_positive_overlay.png"),
            title=f"Attack - Clean Positive Shift L{layer_id} H{head_id}{title_score}",
            alpha=alpha,
        )

        print(f"Saved spatial overlays for L{layer_id:02d}_H{head_id:02d} to {head_dir}")