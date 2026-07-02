import argparse
import numpy as np
from tqdm import tqdm
from utils import open_config, create_model, data_preparation
from detector.utils import process_attn
from visualizor.layer_head_heatmap_visualization import *
from visualizor.token_attention_shift_visualization import *
from visualizor.spatial_attention_overlay_visualization import *

def find_pos_div_index(diff_map_mean, diff_map_std, n=2):
    pos_heads = (diff_map_mean -  n * diff_map_std) > 0
    indices = np.where(pos_heads)
    index_pairs = [list(pair) for pair in zip(indices[0], indices[1])]
    print(f"pos index: {len(index_pairs)}, total: {diff_map_mean.shape[0]*diff_map_mean.shape[1]}")
    
    return index_pairs

def find_top_div_index(diff_map_mean, diff_map_std, portion=0.1):
    pos_heads = diff_map_mean - 1 * diff_map_std
    flattened_pos_heads = pos_heads.flatten()
    total_heads = len(flattened_pos_heads)
    top_n = max(int(portion * total_heads), 1)
    top_indices = np.argpartition(flattened_pos_heads, -top_n)[-top_n:]
    top_index_pairs = [list(np.unravel_index(idx, pos_heads.shape)) for idx in top_indices]

    return top_index_pairs

def find_topk_sensitive_heads(score_map, k=10):
    """
    score_map: shape [num_layers, num_heads]

    Larger score means more sensitive.
    Here, score_map = clean_attn - attack_attn means
    attack increases relative attention to image tokens.
    """
    flat = score_map.flatten()
    idxs = np.argsort(flat)[-k:][::-1]

    heads = []
    for idx in idxs:
        layer, head = np.unravel_index(idx, score_map.shape)
        heads.append((layer, head, float(score_map[layer, head])))

    return heads

def main(args):
    save_dir = "./results/attention_map/instruction_3/"

    # instruction = "Say xxxxxx" # instruction 1
    # instruction = "Please describe the object in this image." # instruction 2
    instruction = "Please describe the image." # instruction 3

    normal_data, attack_data = data_preparation(args)

    model_config_path = f"./configs/model_configs/{args.model_name}_config.json"
    model_config = open_config(config_path=model_config_path)
    model_config["params"]["max_output_tokens"] = 1
    
    if args.use_mm:
        model = create_model(config=model_config, instruction=instruction, image=normal_data)
    else:
        model = create_model(config=model_config)

    model.print_model_info()
    
    access_maps = []
    attack_maps = []

# for data in tqdm(normal_data):
    # _, _, attention_maps, _, input_range, _ = model.inference(instruction, normal_data)
    _, _, attention_maps, _, input_range, _ = model.inference(instruction, attack_data, noise_step=0.5)
    access_attn = process_attn(attention_maps[0], input_range, "normalize_sum")
    access_maps.append(access_attn)

# for data in tqdm(attack_data):
    _, _, attack_attention_maps, _, attack_input_range, _ = model.inference(instruction, attack_data)
    attack_attn = process_attn(attack_attention_maps[0], attack_input_range, "normalize_sum")
    attack_maps.append(attack_attn)
    
    if args.is_vis:
        # 1. Layer × Head heatmap
        heatmap_visualize(access_attn, attack_attn, save_dir=save_dir+"attn_heatmaps")

        # 2. attack-sensitive heads:
        # positive value means attack decreases instruction ratio,
        # i.e., relatively increases image attention.
        image_shift = access_attn - attack_attn

        selected_heads = find_topk_sensitive_heads(image_shift, k=10)

        print("\nSelected attack-sensitive heads:")
        for layer, head, score in selected_heads:
            print(f"Layer {layer}, Head {head}: {score:.6f}")

        plot_selected_layer_head_token_shift(
            clean_attention=attention_maps[0],
            attack_attention=attack_attention_maps[0],
            selected_heads=selected_heads,
            clean_input_range=input_range,
            attack_input_range=attack_input_range,
            save_dir=save_dir+"selected_head_token_attention",
        )

        # 3. save overlay on original clean/attack images
        save_selected_head_spatial_overlay(
            clean_attention=attention_maps[0],
            attack_attention=attack_attention_maps[0],
            clean_image=normal_data,
            attack_image=attack_data,
            clean_input_range=input_range,
            attack_input_range=attack_input_range,
            selected_heads=selected_heads,
            save_dir=save_dir+"selected_head_spatial_overlay",
            alpha=0.45,
            percentile=99.5,
        )

        exit(0)

    access_maps = np.array(access_maps)
    attack_maps = np.array(attack_maps)

    access_mean_maps = np.mean(access_maps, axis=0)
    access_std_maps = np.std(access_maps, axis=0)

    atk_mean_maps = np.mean(attack_maps, axis=0)
    atk_std_maps = np.std(attack_maps, axis=0)
    
    diff_map_mean = access_mean_maps - atk_mean_maps
    diff_map_std = 1 * (access_std_maps + atk_std_maps)
    
    print("Testing dataset: ", args.dataset)
    print("Testing model: ", args.model_name)
    
    for i in range(6):
        print(f"======== index pos (n={i}) =========")
        pos_index_div = find_pos_div_index(diff_map_mean, diff_map_std, n=i)
        print(pos_index_div)
        print(f"propotion: {len(pos_index_div)} ({len(pos_index_div)/(diff_map_mean.shape[0]*diff_map_mean.shape[1])})")
        
    # for i in [0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]:
    #     print(f"======== index pos (n={i}) =========")
    #     pos_index_div = find_top_div_index(diff_map_mean, diff_map_std, portion=i)
    #     print(pos_index_div)
    #     print(f"propotion: {len(pos_index_div)} ({len(pos_index_div)/(diff_map_mean.shape[0]*diff_map_mean.shape[1])})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Open Prompt Injection Experiments')
    parser.add_argument('--model_name', default='qwen2-attn', type=str)
    parser.add_argument('--num_data', default=10, type=int)
    parser.add_argument('--select_index', default="0", type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--use_mm', action="store_true")
    parser.add_argument('--is_vis', action="store_true")
    parser.add_argument('--clean_image_dir', default="./data/clean_images", type=str)

    args = parser.parse_args()

    main(args)