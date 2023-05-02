import time
import torch
from tqdm import trange
from einops import repeat
from lvdm.samplers.ddim import DDIMSampler
from lvdm.models.ddpm3d import FrameInterpPredLatentDiffusion

# ------------------------------------------------------------------------------------------
def add_conditions_long_video(model, cond, batch_size, 
                              sample_cond_noise_level=None, 
                              mask=None,
                              input_shape=None,
                              T_cond_pred=None,
                              cond_frames=None,
                              ):
    assert(isinstance(cond, dict))
    try:
        device = model.device
    except:
        device = next(model.parameters()).device
    
    # add cond noisy level
    if sample_cond_noise_level is not None:
        if getattr(model, "noisy_cond", False):
            assert(sample_cond_noise_level is not None)
            s = sample_cond_noise_level
            s = repeat(torch.tensor([s]), '1 -> b', b=batch_size)
            s = s.to(device).long()
        else:
            s = None
        cond['s'] =s
    
    # add cond mask
    if mask is not None:
        if mask == "uncond":
            mask = torch.zeros(batch_size, 1, *input_shape[2:], device=device)
        elif mask == "pred":
            mask = torch.zeros(batch_size, 1, *input_shape[2:], device=device)
            mask[:, :, :T_cond_pred, :, :] = 1
        elif mask == "interp":
            mask = torch.zeros(batch_size, 1, *input_shape[2:], device=device)
            mask[:, :, 0, :, :] = 1
            mask[:, :, -1, :, :] = 1
        else:
            raise NotImplementedError
        cond['mask'] = mask
        
    # add cond_frames
    if cond_frames is not None:
        if sample_cond_noise_level is not None:
            noise = torch.randn_like(cond_frames)
            noisy_cond = model.q_sample(x_start=cond_frames, t=s, noise=noise)
        else:
            noisy_cond = cond_frames
        cond['noisy_cond'] = noisy_cond

    return cond

# ------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_clip(model, shape, sample_type="ddpm", ddim_steps=None, eta=1.0, cond=None,
                uc=None, uncond_scale=1., uc_type=None, **kwargs):
    
    log = dict()

    # sample batch
    with model.ema_scope("EMA"):
        t0 = time.time()
        if sample_type == "ddpm":
            samples = model.p_sample_loop(cond, shape, return_intermediates=False, verbose=False,
                                          unconditional_guidance_scale=uncond_scale, 
                                          unconditional_conditioning=uc, uc_type=uc_type, )
        elif sample_type == "ddim":
            ddim = DDIMSampler(model)
            samples, intermediates = ddim.sample(S=ddim_steps, batch_size=shape[0], shape=shape[1:], 
                                                 conditioning=cond, eta=eta, verbose=False,
                                                 unconditional_guidance_scale=uncond_scale,
                                                 unconditional_conditioning=uc, uc_type=uc_type,)
    
    t1 = time.time()
    log["sample"] = samples
    log["time"] = t1 - t0
    log['throughput'] = samples.shape[0] / (t1 - t0)
    return log

# ------------------------------------------------------------------------------------------
@torch.no_grad()
def autoregressive_pred(model, batch_size, *args, 
                        T_cond=1, 
                        n_pred_steps=3, 
                        sample_cond_noise_level=None, 
                        decode_single_video_allframes=False,
                        uncond_scale=1.0, 
                        uc_type=None, 
                        max_z_t=None,
                        overlap_t=0,
                        **kwargs):
    
    model.sample_cond_noise_level = sample_cond_noise_level

    image_size = model.image_size
    image_size = [image_size, image_size] if isinstance(image_size, int) else image_size
    C = model.model.diffusion_model.in_channels-1 if isinstance(model, FrameInterpPredLatentDiffusion) \
        else model.model.diffusion_model.in_channels
    T = model.model.diffusion_model.temporal_length
    shape = [batch_size, C, T, *image_size]

    t0 = time.time()
    long_samples = []

    # uncond sample
    log = dict()

    # --------------------------------------------------------------------
    # make condition
    cond = add_conditions_long_video(model, {}, batch_size, mask="uncond", input_shape=shape)
    if (uc_type is None and uncond_scale != 1.0) or (uc_type == 'cfg_original' and uncond_scale != 0.0):
        print('Use Uncondition guidance')
        uc = add_conditions_long_video(model, {}, batch_size, mask="uncond", input_shape=shape)
    else:
        print('NO Uncondition guidance')
        uc=None
    
    # sample an initial clip (unconditional)
    sample = sample_clip(model, shape, cond=cond, 
                         uc=uc, uncond_scale=uncond_scale, uc_type=uc_type,
                         **kwargs)['sample']
    long_samples.append(sample.cpu())

    # extend
    for i in range(n_pred_steps):
        T = sample.shape[2]
        cond_z0 = sample[:, :, T-T_cond:, :, :]
        assert(cond_z0.shape[2] == T_cond)
        
        # make prediction model's condition
        cond = add_conditions_long_video(model, {}, batch_size, mask="pred", input_shape=shape, cond_frames=cond_z0, sample_cond_noise_level=sample_cond_noise_level)
        ## unconditional_guidance's condition
        if (uc_type is None and uncond_scale != 1.0) or (uc_type == 'cfg_original' and uncond_scale != 0.0):
            print('Use Uncondition guidance')
            uc = add_conditions_long_video(model, {}, batch_size, mask="uncond", input_shape=shape)
        else:
            print('NO Uncondition guidance')
            uc=None
        
        # sample a short clip (condition on previous latents)
        sample = sample_clip(model, shape, *args, cond=cond, uc=uc, uncond_scale=uncond_scale, uc_type=uc_type, 
                          **kwargs)['sample']
        ext = sample[:, :, T_cond:, :, :]
        assert(ext.dim() == 5)
        long_samples.append(ext.cpu())
        progress = (i+1)/n_pred_steps * 100
        print(f"Finish pred step {int(progress)}% [{i+1}/{n_pred_steps}]")
    torch.cuda.empty_cache()
    long_samples = torch.cat(long_samples, dim=2)

    t1 = time.time()
    log["sample"] = long_samples
    log["time"] = t1 - t0
    log['throughput'] = long_samples.shape[0] / (t1 - t0)
    return log


# ------------------------------------------------------------------------------------------
@torch.no_grad()
def interpolate(base_samples, 
                model, batch_size, *args, sample_video=False,
                sample_cond_noise_level=None, decode_single_video_allframes=False,
                uncond_scale=1.0, uc_type=None, max_z_t=None,
                overlap_t=0, prompt=None, config=None, 
                interpolate_cond_fps=None,
                **kwargs):

    model.sample_cond_noise_level = sample_cond_noise_level
    
    N, c, t, h, w = base_samples.shape
    n_steps = len(range(0, t-3, 3))
    device = next(model.parameters()).device
    if N < batch_size:
        batch_size = N
    elif N > batch_size:
        raise ValueError
    assert(N == batch_size)

    C = model.model.diffusion_model.in_channels-1 if isinstance(model, FrameInterpPredLatentDiffusion) and model.concat_mask_on_input \
        else model.model.diffusion_model.in_channels
    image_size = model.image_size
    image_size = [image_size, image_size] if isinstance(image_size, int) else image_size
    T = model.model.diffusion_model.temporal_length
    shape = [batch_size, C, T, *image_size ]

    t0 = time.time()
    long_samples = []
    cond = {}
    for i in trange(n_steps, desc='Interpolation Steps'):
        cond_z0 = base_samples[:, :, i:i+2, :, :].cuda()
        # make prediction model's condition
        cond = add_conditions_long_video(model, {}, batch_size, mask="interp", input_shape=shape, cond_frames=cond_z0, sample_cond_noise_level=sample_cond_noise_level)
        ## unconditional_guidance's condition
        if (uc_type is None and uncond_scale != 1.0) or (uc_type == 'cfg_original' and uncond_scale != 0.0):
            print('Use Uncondition guidance')
            uc = add_conditions_long_video(model, {}, batch_size, mask="uncond", input_shape=shape)
        else:
            print('NO Uncondition guidance')
            uc=None
        
        # sample an interpolation clip
        sample = sample_clip(model, shape, *args, cond=cond, uc=uc, 
                          uncond_scale=uncond_scale, 
                          uc_type=uc_type, 
                          **kwargs)['sample']
        ext = sample[:, :, 1:-1, :, :]
        assert(ext.dim() == 5)
        assert(ext.shape[2] == T - 2)
        # -----------------------------------------------
        if i != n_steps - 1:
            long_samples.extend([cond_z0[:, :, 0:1, :, :].cpu(), ext.cpu()])
        else:
            long_samples.extend([cond_z0[:, :, 0:1, :, :].cpu(), ext.cpu(), cond_z0[:, :, 1:, :, :].cpu()])
        # -----------------------------------------------
    
    torch.cuda.empty_cache()
    long_samples = torch.cat(long_samples, dim=2)
    
    # decode
    long_samples_decoded = []
    print('Decoding ...')
    for i in trange(long_samples.shape[0]):
        torch.cuda.empty_cache()
        
        long_sample = long_samples[i].unsqueeze(0).cuda()
        if overlap_t != 0:
            print('Use overlapped decoding')
            res = model.overlapped_decode(long_sample, max_z_t=max_z_t, 
                                        overlap_t=overlap_t).cpu()
        else:
            res = model.decode_first_stage(long_sample, 
                                    bs=None, 
                                    decode_single_video_allframes=decode_single_video_allframes,
                                    max_z_t=max_z_t).cpu()
        long_samples_decoded.append(res)
        torch.cuda.empty_cache()

    long_samples_decoded = torch.cat(long_samples_decoded, dim=0)

    # log
    t1 = time.time()
    log = {}
    log["sample"] = long_samples_decoded
    log["sample_z"] = long_samples
    log["time"] = t1 - t0
    torch.cuda.empty_cache()
    return log
