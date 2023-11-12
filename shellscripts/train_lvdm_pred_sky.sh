export PATH=/apdcephfs_cq2/share_1290939/yingqinghe/anaconda/envs/ldmA100/bin/:$PATH
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"

export TOKENIZERS_PARALLELISM=false

PROJ_ROOT="/apdcephfs_cq2/share_1290939/yingqinghe/results/latent_diffusion"
EXPNAME="test_sky_train_pred"
CONFIG="configs/lvdm_long/sky_pred.yaml"
DATADIR="/dockerdata/sky_timelapse"
AEPATH="/apdcephfs/share_1290939/yingqinghe/results/latent_diffusion/LVDM/ae_013_sky256_basedon003_4nodes_e0/checkpoints/trainstep_checkpoints/epoch=000299-step=000010199.ckpt"

# run
python main.py \
--base $CONFIG \
-t --gpus 0, \
--name $EXPNAME \
--logdir $PROJ_ROOT \
--auto_resume True \
lightning.trainer.num_nodes=1 \
data.params.train.params.data_root=$DATADIR \
data.params.validation.params.data_root=$DATADIR \
model.params.first_stage_config.params.ckpt_path=$AEPATH


# commands for multi nodes training
# ---------------------------------------------------------------------------------------------------
# python -m torch.distributed.run \
# --nproc_per_node=8 --nnodes=$NHOST --master_addr=$MASTER_ADDR --master_port=1234 --node_rank=$INDEX \
# main.py \
# --base $CONFIG \
# -t --gpus 0,1,2,3,4,5,6,7 \
# --name $EXPNAME \
# --logdir $PROJ_ROOT \
# --auto_resume True \
# lightning.trainer.num_nodes=$NHOST \
# data.params.train.params.data_root=$DATADIR \
# data.params.validation.params.data_root=$DATADIR
