
CKPT_PRED="models/lvdm_long/sky_pred.ckpt"
CKPT_INTERP="models/lvdm_long/sky_interp.ckpt"
AEPATH="models/ae/ae_sky.ckpt"
CONFIG_PRED="configs/lvdm_long/sky_pred.yaml"
CONFIG_INTERP="configs/lvdm_long/sky_interp.yaml"
OUTDIR="results/longvideos/"

python scripts/sample_uncond_long_videos.py \
    --ckpt_pred $CKPT_PRED \
    --config_pred $CONFIG_PRED \
    --ckpt_interp $CKPT_INTERP \
    --config_interp $CONFIG_INTERP \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1000 \
    --show_denoising_progress \
    model.params.first_stage_config.params.ckpt_path=$AEPATH \
    --sample_cond_noise_level 100 \
    --uncond_scale 0.1 \
    --n_pred_steps 2 \
    --sample_type ddim --ddim_steps 50

# if use DDPM： remove: `--sample_type ddim --ddim_steps 50`