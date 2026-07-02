import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_selected_layer_head_token_shift(
    clean_attention,
    attack_attention,
    selected_heads,
    clean_input_range,
    attack_input_range,
    save_dir="./selected_head_token_attention",
):
    """
    Plot token-level attention shift for selected layer-head pairs.

    selected_heads:
        list of (layer_id, head_id, score) or (layer_id, head_id)
    """
    os.makedirs(save_dir, exist_ok=True)

    for item in selected_heads:
        if len(item) == 3:
            layer_id, head_id, score = item
        else:
            layer_id, head_id = item
            score = None

        clean_vec = clean_attention[layer_id].to(torch.float32).cpu().numpy()[0, head_id, -1, :]
        attack_vec = attack_attention[layer_id].to(torch.float32).cpu().numpy()[0, head_id, -1, :]

        seq_len = min(len(clean_vec), len(attack_vec))
        clean_vec = clean_vec[:seq_len]
        attack_vec = attack_vec[:seq_len]
        diff = attack_vec - clean_vec

        x = np.arange(seq_len)

        head_dir = os.path.join(save_dir, f"L{layer_id:02d}_H{head_id:02d}")
        os.makedirs(head_dir, exist_ok=True)

        inst_range, image_range = clean_input_range

        # shared y scale for clean vs attack
        ymax = np.percentile(np.concatenate([clean_vec, attack_vec]), 99.5)
        if ymax <= 0:
            ymax = max(clean_vec.max(), attack_vec.max()) + 1e-8

        title_score = "" if score is None else f" | score={score:.6f}"

        plt.figure(figsize=(18, 5))
        plt.plot(x, clean_vec, label="Clean")
        plt.plot(x, attack_vec, label="Attack")

        if inst_range is not None:
            plt.axvline(inst_range[0], linestyle="--", linewidth=1)
            plt.axvline(inst_range[1], linestyle="--", linewidth=1)

        if image_range is not None:
            plt.axvline(image_range[0], linestyle="--", linewidth=1)
            plt.axvline(image_range[1], linestyle="--", linewidth=1)

        plt.ylim(0, ymax)
        plt.xlabel("Input token position")
        plt.ylabel("Attention weight")
        plt.title(f"Layer {layer_id}, Head {head_id}: Clean vs Attack Token Attention{title_score}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(head_dir, "clean_vs_attack_token_attention.png"), dpi=300)
        plt.close()

        # diff plot
        max_abs = np.percentile(np.abs(diff), 99.5)
        if max_abs <= 0:
            max_abs = np.max(np.abs(diff)) + 1e-8

        plt.figure(figsize=(18, 5))
        plt.plot(x, diff, label="Attack - Clean")
        plt.axhline(0, linewidth=1)

        if inst_range is not None:
            plt.axvline(inst_range[0], linestyle="--", linewidth=1)
            plt.axvline(inst_range[1], linestyle="--", linewidth=1)

        if image_range is not None:
            plt.axvline(image_range[0], linestyle="--", linewidth=1)
            plt.axvline(image_range[1], linestyle="--", linewidth=1)

        plt.ylim(-max_abs, max_abs)
        plt.xlabel("Input token position")
        plt.ylabel("Attention shift")
        plt.title(f"Layer {layer_id}, Head {head_id}: Attack - Clean Token Shift{title_score}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(head_dir, "attack_minus_clean_token_shift.png"), dpi=300)
        plt.close()

        print(f"Saved selected head L{layer_id:02d}_H{head_id:02d} to {head_dir}")