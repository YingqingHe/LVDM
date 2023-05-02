import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time
import math
import argparse
import numpy as np
from tqdm import trange
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
from pytorch_lightning import seed_everything

from lvdm.samplers.ddim import DDIMSampler
from lvdm.utils.common_utils import torch_to_np, str2bool
from lvdm.utils.dist_utils import setup_dist, gather_data
from scripts.sample_utils import (
    load_model, 
    save_args, 
    make_model_input_shape, 
    sample_batch,
    save_results, 
    )

# ------------------------------------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser()
    # basic args
    parser.add_argument("--ckpt_path", type=str, help="model checkpoint path")
    parser.add_argument("--config_path", type=str, help="model config path (a yaml file)")
    parser.add_argument("--prompt", type=str, help="input text prompts for text2video (a sentence OR a txt file).")
    parser.add_argument("--save_dir", type=str, help="results saving dir", default="results/")
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
    parser.add_argument("--uncond_scale", type=float, default=15.0, help="uncondition guidance scale")
    # saving args
    parser.add_argument("--save_mp4", type=str2bool, default=True, help="whether save samples in separate mp4 files", choices=["True", "true", "False", "false"])
    parser.add_argument("--save_mp4_sheet", action='store_true', default=False, help="whether save samples in mp4 file",)
    parser.add_argument("--save_npz", action='store_true', default=False, help="whether save samples in npz file",)
    parser.add_argument("--save_jpg", action='store_true', default=False, help="whether save samples in jpg file",)
    parser.add_argument("--save_fps", type=int, default=8, help="fps of saved mp4 videos",)
    return parser

# ------------------------------------------------------------------------------------------
@torch.no_grad()
def sample(model, noise_shape, n_iters, ddp=False, **kwargs):
    all_videos = []
    for _ in trange(n_iters, desc="Sampling Batches (unconditional)"):
        samples = sample_batch(model, noise_shape, condition=None, **kwargs)
        samples = model.decode_first_stage(samples)
        if ddp: # gather samples from multiple gpus
            data_list = gather_data(samples, return_np=False)
            all_videos.extend([torch_to_np(data) for data in data_list])
        else:
            all_videos.append(torch_to_np(samples))
    all_videos = np.concatenate(all_videos, axis=0)
    return all_videos

# ------------------------------------------------------------------------------------------
def main():
    """
    unconditional generation of short videos
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
    config = OmegaConf.load(opt.config_path)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)
    print("config: \n", config)

    # get model & sampler
    model, _, _ = load_model(config, opt.ckpt_path)
    ddim_sampler = DDIMSampler(model) if opt.sample_type == "ddim" else None

    # sample
    start = time.time()
    noise_shape = make_model_input_shape(model, opt.batch_size, T=opt.num_frames)
    ngpus = 1 if not opt.ddp else dist.get_world_size()
    n_iters = math.ceil(opt.n_samples / (ngpus * opt.batch_size))
    samples = sample(model, noise_shape, n_iters, sampler=ddim_sampler, **vars(opt))
    assert(samples.shape[0] >= opt.n_samples)
    
    # save
    if (opt.ddp and dist.get_rank() == 0) or (not opt.ddp):
        if opt.seed is not None:
            save_name = f"seed{seed:05d}"
        save_results(samples, opt.save_dir, save_name=save_name, save_fps=opt.save_fps)
    print("Finish sampling!")
    print(f"Run time = {(time.time() - start):.2f} seconds")

    if opt.ddp:
        dist.destroy_process_group()

# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()