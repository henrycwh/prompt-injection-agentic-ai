# Prompt Injection in Agentic AI

This repository is adapted from **Attention Tracker: Detecting Prompt Injection Attacks in LLMs** and extends the original codebase for multimodal prompt injection experiments with adversarial images.

The original Attention Tracker repository provides scripts and tools to identify important attention heads and evaluate prompt injection attacks on large language models (LLMs). This version keeps the original text-only pipeline and adds multimodal analysis using image inputs.

Original project page: https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker  
Original paper: https://arxiv.org/abs/2411.00348  

---

## Features

### Original Attention Tracker Features

- **Identify Important Heads**: Determine which attention heads are critical for detecting prompt injection attacks.
- **Run Experiments**: Execute experiments on datasets to evaluate the model's effectiveness.
- **Test Queries**: Test individual queries against your chosen model.

### Added in This Version

- **Multimodal Mode**: Add `--use_mm` for image-based prompt injection experiments.
- **LLaVA-NeXT Support**: Add `MultimodalAttentionModel` in `models/mm_attn_model.py`.
- **Clean vs. Attack Image Analysis**: Compare attention patterns between clean and adversarial images.
- **Visualization Tools**: Add `--is_vis` for layer-head heatmaps, token attention shifts, and spatial attention overlays.

---

## Main Modifications

The main extension is the `--use_mm` argument.

When `--use_mm` is enabled, the code switches from text-only prompt injection data to multimodal image-based data. This affects:

- `data_preparation(args)` in `utils.py`
- `create_model(config, instruction, image)` in `utils.py`
- `MultimodalAttentionModel` in `models/mm_attn_model.py`

The multimodal setting currently compares a clean image and an adversarial prompt image, then analyzes how attention shifts between instruction tokens and image tokens.

Visualization can be enabled with:

```bash
--is_vis
```

This saves attention heatmaps, token-level attention shift plots, and spatial attention overlays under:

```text
results/attention_map/
```

---

## Getting Started

### Prerequisites

Ensure Python is installed, then install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Find Important Heads

```bash
./scripts/find_heads.sh
```

### Specify Important Heads

Edit the following file:

```text
configs/model_configs/{model_name}.json
```

Then modify:

```json
["params"]["important_heads"]
```

### Run Experiments on DeepSet Prompt Injection Dataset

```bash
./scripts/run_dataset.sh
```
Injected Image Dataset: https://drive.google.com/drive/folders/17MDXgdW_jdFxIlNX29nzt0VLK7PzV7wb?usp=drive_link
Attention Tracker Dataset: https://huggingface.co/datasets/deepset/prompt-injections?row=19

### Test Individual Queries

```bash
python run.py --model_name {model} --test_query "{query you want to test}"
```

### Run Multimodal Prompt Injection Experiment

```bash
python select_head.py --model_name qwen2-attn --use_mm
```

### Run Multimodal Experiment with Visualization

```bash
python select_head.py --model_name qwen2-attn --use_mm --is_vis
```

---

## Important Files

```text
select_head.py                  Main script for attention analysis
utils.py                        Config loading, model creation, and data preparation
models/mm_attn_model.py          Multimodal attention model based on LLaVA-NeXT
visualizor/                     Attention visualization tools
configs/model_configs/           Model configuration files
scripts/                        Experiment scripts
```

---

## License

This project is based on the original Attention Tracker repository.

Original license: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en)

---

## Citation

If you use the original Attention Tracker method, please cite:

```bibtex
@misc{hung2024attentiontrackerdetectingprompt,
      title={Attention Tracker: Detecting Prompt Injection Attacks in LLMs}, 
      author={Kuo-Han Hung and Ching-Yun Ko and Ambrish Rawat and I-Hsin Chung and Winston H. Hsu and Pin-Yu Chen},
      year={2024},
      eprint={2411.00348},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2411.00348}, 
}
```
