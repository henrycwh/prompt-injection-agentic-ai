MODELS=( "llama3-llava-next-8b-attn")
for MODEL in "${MODELS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 -u select_head.py \
                            --model_name ${MODEL} \
                            --use_mm \
                            --dataset llm  >> "analysis.txt" \
                            --is_vis
done
