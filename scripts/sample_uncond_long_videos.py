import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import math
import torch
import time
import argparse
import numpy as np
from tqdm import trange
from omegaconf import OmegaConf
import torch.distributed as dist
from pytorch_lightning import seed_everything

from lvdm.utils.dist_utils import setup_dist, gather_data
from lvdm.utils.common_utils import torch_to_np, str2bool
from scripts.sample_long_videos_utils import autoregressive_pred, interpolate
from scripts.sample_utils import load_model, save_args, save_results

def get_parser():
    parser = argparse.ArgumentParser()
    # basic args
    parser.add_argument("--ckpt_pred", type=str, default="", help="ckpt path")
    parser.add_argument("--ckpt_interp", type=str, default=None, help="ckpt path")
    parser.add_argument("--config_pred", type=str, default="", help="config yaml")
    parser.add_argument("--config_interp", type=str, default=None, help="config yaml")
    parser.add_argument("--save_dir", type=str, default="results/longvideos", help="results saving dir")
    # device args
    parser.add_argument("--ddp", action='store_true', help="whether use pytorch ddp mode for parallel sampling (recommend for multi-gpu case)", default=False)
    parser.add_argument("--local_rank", type=int, help="is used for pytorch ddp mode", default=0)
    parser.add_argument("--gpu_id", type=int, help="choose a specific gpu", default=0)
    # sampling args
    parser.add_argument("--n_samples", type=int, help="how many samples for each text prompt", default=2)
    parser.add_argument("--batch_size", type=int, help="video batch size for sampling", default=1)
    parser.add_argument("--sample_type", type=str, help="ddpm or ddim", default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, help="ddim sampling -- number of ddim denoising timesteps", default=50)
    parser.add_argument("--eta", type=float, help="ddim sampling -- eta (0.0 yields deterministic sampling, 1.0 yields random sampling)", default=1.0)
    parser.add_argument("--seed", type=int, default=None, help="fix a seed for randomness (If you want to reproduce the sample results)")
    parser.add_argument("--num_frames", type=int, default=None, help="number of input frames")
    parser.add_argument("--show_denoising_progress", action='store_true', default=False, help="whether show denoising progress during sampling one batch",)
    parser.add_argument("--uncond_scale", type=float, default=1.0, help="unconditional guidance scale")
    parser.add_argument("--uc_type", type=str, help="unconditional guidance scale", default="cfg_original", choices=["cfg_original", None])
    # prediction & interpolation args
    parser.add_argument("--T_cond", type=int, default=1, help="temporal length of condition frames")
    parser.add_argument("--n_pred_steps", type=int, default=None, help="")
    parser.add_argument("--sample_cond_noise_level", type=int, default=None, help="")
    parser.add_argument("--overlap_t", type=int, default=0, help="")
    # saving args
    parser.add_argument("--save_mp4", type=str2bool, default=True, help="whether save samples in separate mp4 files", choices=["True", "true", "False", "false"])
    parser.add_argument("--save_mp4_sheet", action='store_true', default=False, help="whether save samples in mp4 file",)
    parser.add_argument("--save_npz", action='store_true', default=False, help="whether save samples in npz file",)
    parser.add_argument("--save_jpg", action='store_true', default=False, help="whether save samples in jpg file",)
    parser.add_argument("--save_fps", type=int, default=8, help="fps of saved mp4 videos",)
    return parser

# ------------------------------------------------------------------------------------------
@torch.no_grad()
def sample(model_pred, model_interp, n_iters, ddp=False, all_gather=False, **kwargs):
    all_videos = []
    for _ in trange(n_iters, desc="Sampling Batches (unconditional)"):

        # autoregressive predict latents
        logs = autoregressive_pred(model_pred, **kwargs)
        samples_z = logs['sample'] if isinstance(logs, dict) else logs
        
        # interpolate latents
        logs = interpolate(samples_z, model_interp, **kwargs)
        samples=logs['sample']
        
        if ddp and all_gather: # gather samples from multiple gpus
            data_list = gather_data(samples, return_np=False)
            all_videos.extend([torch_to_np(data) for data in data_list])
        else:
            all_videos.append(torch_to_np(samples))
    all_videos = np.concatenate(all_videos, axis=0)
    return all_videos

# ------------------------------------------------------------------------------------------
def main():
    """
    unconditional generation of long videos
    """
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    os.makedirs(opt.save_dir, exist_ok=True)
    save_args(opt.save_dir, opt)
    
    # set device
    if opt.ddp:
        setup_dist(opt.local_rank)
        opt.n_samples = math.ceil(opt.n_samples / dist.get_world_size())
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{opt.gpu_id}"
    
    # set random seed
    if opt.seed is not None:
        seed = opt.local_rank + opt.seed if opt.ddp else opt.seed
        seed_everything(seed)
    
    # load & merge config
    config_pred = OmegaConf.load(opt.config_pred)
    cli = OmegaConf.from_dotlist(unknown)
    config_pred = OmegaConf.merge(config_pred, cli)
    if opt.config_interp is not None:
        config_interp = OmegaConf.load(opt.config_interp)
        cli = OmegaConf.from_dotlist(unknown)
        config_interp = OmegaConf.merge(config_interp, cli)
    
    # calculate n_pred_steps
    if opt.num_frames is not None:
        temporal_down = config_pred.model.params.first_stage_config.params.ddconfig.encoder.params.downsample[0]
        temporal_length_z = opt.num_frames // temporal_down
        model_pred_length = config_pred.model.params.unet_config.params.temporal_length
        if opt.config_interp is not None:
            model_interp_length = config_interp.model.params.unet_config.params.temporal_length - 2
            pred_length =  math.ceil((temporal_length_z + model_interp_length) / (model_interp_length + 1))
        else:
            pred_length = temporal_length_z
        n_pred_steps = math.ceil((pred_length - model_pred_length) / (model_pred_length - 1))
        opt.n_pred_steps = n_pred_steps
        print(f'Temporal length {opt.num_frames} needs latent length = {temporal_length_z}; \n \
                pred_length = {pred_length}; \n \
                prediction steps = {n_pred_steps}')
    else:
        assert(opt.n_pred_steps is not None)
    
    # model
    model_pred, _, _ = load_model(config_pred, opt.ckpt_pred)
    model_interp, _, _ = load_model(config_interp, opt.ckpt_interp)
    
    # sample
    start = time.time()
    ngpus = 1 if not opt.ddp else dist.get_world_size()
    n_iters = math.ceil(opt.n_samples / (ngpus * opt.batch_size))
    samples = sample(model_pred, model_interp, n_iters, **vars(opt))
    assert(samples.shape[0] >= opt.n_samples)
    
    # save
    if (opt.ddp and dist.get_rank() == 0) or (not opt.ddp):
        if opt.seed is not None:
            save_name = f"seed{seed:05d}"
        save_results(samples, opt.save_dir, save_name=save_name, save_fps=opt.save_fps)
    print("Finish sampling!")
    print(f"total time = {int(time.time()- start)} seconds, \
          num of iters = {n_iters}; \
          num of samples = {ngpus * opt.batch_size * n_iters}; \
          temporal length = {opt.num_frames}")

    if opt.ddp:
        dist.destroy_process_group()

# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()