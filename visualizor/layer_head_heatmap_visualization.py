import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_layer_head_heatmap(mat, title, save_path=None):
    """
    mat: shape [num_layers, num_heads]
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Attention ratio")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def heatmap_visualize(access_attn, attack_attn, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    diff_attn = attack_attn - access_attn
    image_shift = access_attn - attack_attn

    # clean / attack 建議共用 color scale
    vmin = min(access_attn.min(), attack_attn.min())
    vmax = max(access_attn.max(), attack_attn.max())

    plot_layer_head_heatmap(
        access_attn,
        "Clean Image: Instruction Attention Ratio",
        os.path.join(save_dir, "clean_instruction_ratio.png")
    )

    plot_layer_head_heatmap(
        attack_attn,
        "Attack Image: Instruction Attention Ratio",
        os.path.join(save_dir, "attack_instruction_ratio.png")
    )

    plot_layer_head_heatmap(
        diff_attn,
        "Attack - Clean: Instruction Attention Ratio Change",
        os.path.join(save_dir, "attack_minus_clean_instruction_ratio.png")
    )

    plot_layer_head_heatmap(
        image_shift,
        "Clean - Attack: Image Attention Increase under Attack",
        os.path.join(save_dir, "attack_image_shift.png")
    )

    print("Saved layer-head heatmaps to:", save_dir)