
import os
import sys
import time
import math
import json
import argparse
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

from lvdm.utils.common_utils import instantiate_from_config, shift_dim
from scripts.fvd_utils.fvd_utils import get_fvd_logits, frechet_distance, load_fvd_model, polynomial_mmd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_path', type=str, default='fake data path')
    parser.add_argument('--real_path', type=str, default='real data dir')
    parser.add_argument('--yaml', type=str, default=None, help="training config file for construct dataloader")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--n_runs', type=int, default=1, help='calculate multiple times')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--res_dir', type=str, default='')
    parser.add_argument('--n_sample', type=int, default=2048)
    parser.add_argument('--start_clip_id', type=int, default=None, help="for the evaluation of long video generation")
    parser.add_argument('--end_clip_id', type=int, default=None, help="for the evaluation of long video generation")
    args = parser.parse_args()
    return args

def run(args, run_id, fake_path, i3d, loader, device):
    start = time.time()

    print('load fake videos in numpy ...')
    if fake_path.endswith('npz'):
        fake_data = np.load(fake_path)['arr_0']
    elif fake_path.endswith('npy'):
        fake_data = np.load(fake_path)
    else:
        print(fake_path)
        raise NotImplementedError
    
    s = time.time()
    real_embeddings = []
    for batch in tqdm(loader, desc="Extract Real Embedding", total=math.ceil(args.n_sample / args.batch_size)):
        if len(real_embeddings)*args.batch_size >=args.n_sample: break
        videos = shift_dim((batch[args.image_key]+1)*255/2, 1, -1).int().data.numpy() # convert to 0-255
        real_embeddings.append(get_fvd_logits(videos, i3d=i3d, device=device, batch_size=args.batch_size))
    real_embeddings = torch.cat(real_embeddings, 0)[:args.n_sample]
    t = time.time() - s
    s = time.time()
    fake_embeddings = []
    n_batch = fake_data.shape[0]//args.batch_size
    for i in tqdm(range(n_batch), desc="Extract Fake Embedding"):
        fake_embeddings.append(get_fvd_logits(fake_data[i*args.batch_size:(i+1)*args.batch_size], i3d=i3d, device=device, batch_size=args.batch_size))
    fake_embeddings = torch.cat(fake_embeddings, 0)[:args.n_sample]
    t = time.time() - s
    
    print('calculate fvd ...')
    fvd = frechet_distance(fake_embeddings, real_embeddings)
    fvd = fvd.cpu().numpy() # np float32
    
    print('calculate kvd ...')
    kvd = polynomial_mmd(fake_embeddings.cpu(), real_embeddings.cpu()) # np float 64
    total = time.time() - start

    print(f'Run_id = {run_id}')
    print(f'FVD = {fvd:.2f}')
    print(f'KVD = {kvd:.2f}')
    print(f'Time = {total:.2f}')
    return [fvd, kvd, total]

def run_multitimes(args, i3d, loader, device):
    res_all = []
    for i in range(args.n_runs):
        run_id = i
        res = run(args, run_id, fake_path=args.fake_path, i3d=i3d, loader=loader, device=device)
        res_all.append(np.array(res))
    res_avg = np.mean(np.stack(res_all, axis=0), axis=0)
    res_std = np.std(np.stack(res_all, axis=0), axis=0)
    
    print(f'Results of {args.n_runs} runs:')
    print(f'FVD = {res_avg[0]} ({res_std[0]})')
    print(f'KVD = {res_avg[1]} ({res_std[1]})')
    print(f'Time = {res_avg[2]} ({res_std[2]})')
    
    # dump results
    res={'FVD': f'{res_avg[0]} ({res_std[0]})',
         'KVD': f'{res_avg[1]} ({res_std[1]})',
         'Time': f'{res_avg[2]} ({res_std[2]})',
         'Clip_path': f'{args.fake_path}'
         }
    f = open(os.path.join(args.res_dir, f'{args.n_runs}runs_fvd_stat.json'), 'w')
    json.dump(res, f)
    f.close()

def run_multitimes_dir(args, i3d, loader, device):
    datalist = sorted(os.listdir(args.fake_path))
    for i in range(args.n_runs):
        
        run_id = i

        for idx, path in tqdm(enumerate(datalist)):
            if args.start_clip_id is not None and idx < args.start_clip_id:
                continue
            
            print(f'Cal metrics for clip: {idx}, data: {path}')
            fvd, kvd, total = run(args, run_id, fake_path=os.path.join(args.fake_path, path), 
                                  i3d=i3d, loader=loader, device=device)
            print(f"Run id {run_id}, Clip {idx}, FVD={fvd}, KVD={kvd}, Time={total}")
            
            # dump results
            fvd = float(fvd) if isinstance(fvd, np.ndarray) else fvd
            kvd = float(kvd) if isinstance(kvd, np.ndarray) else kvd

            res={'FVD': fvd, 'KVD': kvd, 'Time': total, 'Clip_path': path}
            f = open(os.path.join(args.res_dir, f'run{run_id}_clip{idx}_{device}.json'), 'w')
            json.dump(res, f)
            f.close()
            
            if args.end_clip_id is not None and idx == args.end_clip_id:
                break
            

if __name__ == '__main__':
    args = get_args()
    print(args)
    os.makedirs(args.res_dir, exist_ok=True)

    print('load i3d ...')
    if args.gpu_id is not None:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cuda')
    i3d = load_fvd_model(device)

    print('prepare dataset and dataloader ...')
    config = OmegaConf.load(args.yaml)
    config.data.params.train.params.data_root=args.real_path
    dataset = instantiate_from_config(config.data.params.train)
    if 'first_stage_key' in config.model.params:
        args.image_key = config.model.params.first_stage_key
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=True)

    # run
    if os.path.isdir(args.fake_path):
        run_multitimes_dir(args, i3d, loader, device)
    else:
        run_multitimes(args, i3d, loader, device)
    
   