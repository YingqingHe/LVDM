
PROMPT="astronaut riding a horse" # OR: PROMPT="input/prompts.txt" for sampling multiple prompts
OUTDIR="results/t2v"

CKPT_PATH="models/t2v/model.ckpt"
CONFIG_PATH="configs/lvdm_short/text2video.yaml"

python scripts/sample_text2video.py \
    --ckpt_path $CKPT_PATH \
    --config_path $CONFIG_PATH \
    --prompt "$PROMPT" \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1000 \
    --show_denoising_progress \
    --save_jpg
