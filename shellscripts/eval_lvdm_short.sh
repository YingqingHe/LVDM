
DATACONFIG="configs/lvdm_short/sky.yaml"
FAKEPATH='${Your_Path}/2048x16x256x256x3-samples.npz'
REALPATH='/dataset/sky_timelapse'
RESDIR='results/fvd'

mkdir -p $res_dir
python scripts/eval_cal_fvd_kvd.py \
    --yaml ${DATACONFIG} \
    --real_path ${REALPATH} \
    --fake_path ${FAKEPATH} \
    --batch_size 32 \
    --num_workers 4 \
    --n_runs 10 \
    --res_dir ${RESDIR} \
    --n_sample 2048
