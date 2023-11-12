
CONFIG_PATH="configs/lvdm_short/sky.yaml"
BASE_PATH="models/lvdm_short/short_sky.ckpt"
AEPATH="models/ae/ae_sky.ckpt"
OUTDIR="results/uncond_short/"

python scripts/sample_uncond.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1000 \
    --show_denoising_progress \
    model.params.first_stage_config.params.ckpt_path=$AEPATH

# if use DDIM： add: `--sample_type ddim --ddim_steps 50`
