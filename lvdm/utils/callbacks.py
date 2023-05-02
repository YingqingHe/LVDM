import logging
import time, os
import numpy as np
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
# ---------------------------------------------------------------------------------
class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 fps=10, log_to_tblogger=True,
                 ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if int((pl.__version__).split('.')[1])>=7:
            self.logger_log_images = {
                pl.loggers.CSVLogger: self._testtube,
            }
        else:
            self.logger_log_images = {
                pl.loggers.TestTubeLogger: self._testtube,
            }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.log_to_tblogger = log_to_tblogger
        self.save_fps = fps


    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        """ log images and videos to tensorboard """

        for k in images:
            tag = f"{split}/{k}"
            if images[k].dim() == 5:
                video = images[k]
                n = video.shape[0]
                video = video.permute(2, 0, 1, 3, 4)
                frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(np.sqrt(n))) for framesheet in video]
                grid = torch.stack(frame_grids, dim=0)
                grid = (grid + 1.0) / 2.0
                grid = grid.unsqueeze(dim=0)
                pl_module.logger.experiment.add_video(
                    tag, grid,
                    global_step=pl_module.global_step)
            else:
                grid = torchvision.utils.make_grid(images[k])
                grid = (grid + 1.0) / 2.0
                pl_module.logger.experiment.add_image(
                    tag, grid,
                    global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx, rank_idx):
        """ save images and videos from images dict """

        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)

        def save_img_grid(grid, path, rescale):
            if rescale:
                    grid = (grid + 1.0) / 2.0
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)
        fps = images.pop('fps', None)
        fs = images.pop('frame_stride', None)

        for k in images:
            img = images[k]
            if isinstance(img, list) and isinstance(img[0], str):
                # a batch of captions
                filename = "string_{}_gs-{:06}_e-{:06}_b-{:06}_r-{:02}.txt".format(
                    k,
                    global_step,
                    current_epoch,
                    batch_idx,
                    rank_idx,
                    )
                path = os.path.join(root, filename)
                with open(path, 'w') as f:
                    for i, txt in enumerate(img):
                        f.write(f'idx={i}, txt={txt}\n')
                f.close()
            elif img.dim() == 5:
                # save video grids
                video = img # b,c,t,h,w
                n = video.shape[0]
                video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
                frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(np.sqrt(n))) for framesheet in video] # [3, grid_h, grid_w]
                grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [T, 3, grid_h, grid_w]
                if self.rescale:
                    grid = (grid + 1.0) / 2.0
                grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, 3, grid_h, grid_w] -> [T, grid_h, grid_w, 3]
                filename = "video_{}_gs-{:06}_e-{:06}_b-{:06}_r-{:02}.mp4".format(
                    k,
                    global_step,
                    current_epoch,
                    batch_idx,
                    rank_idx,
                    )
                filename = filename.split('.mp4')[0] + f'_fps{fps}.mp4' if fps is not None else filename
                filename = filename.split('.mp4')[0] + f'_fs{fs}.mp4' if fs is not None else filename
                path = os.path.join(root, filename)
                print('Save video ...')
                torchvision.io.write_video(path, grid, fps=self.save_fps, video_codec='h264', options={'crf': '10'})
                print('Finish!')
                
                # save frame sheet
                video_frames = rearrange(img, 'b c t h w -> (b t) c h w')
                t = img.shape[2]
                grid = torchvision.utils.make_grid(video_frames, nrow=t)
                filename = "framesheet_{}_gs-{:06}_e-{:06}_b-{:06}_r-{:02}.jpg".format(
                    k,
                    global_step,
                    current_epoch,
                    batch_idx,
                    rank_idx,
                    )
                path = os.path.join(root, filename)
                print('Save framesheet ...')
                save_img_grid(grid, path, self.rescale)
                print('Finish!')
            else:
                grid = torchvision.utils.make_grid(img, nrow=4)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}_r-{:02}.jpg".format(
                    k,
                    global_step,
                    current_epoch,
                    batch_idx,
                    rank_idx,
                    )
                path = os.path.join(root, filename)
                print('Save image grid ...')
                save_img_grid(grid, path, self.rescale)
                print('Finish!')
                
    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train", rank=0):
        """ generate images, then save and log to tensorboard """
        
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        
        if (self.check_frequency(check_idx) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()
                torch.cuda.empty_cache()

            with torch.no_grad():
                log_func = pl_module.log_videos if hasattr(pl_module, 'is_video') and pl_module.is_video else pl_module.log_images
                images = log_func(batch, split=split)
                torch.cuda.empty_cache()

            # process images
            for k in images:
                if hasattr(images[k], 'shape'):
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            
            print("Log local ...")
            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, 
                           batch_idx, rank)
            if self.log_to_tblogger:
                print("Log images to logger ...")
                logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
                logger_log_images(pl_module, images, pl_module.global_step, split)
                print('Finish!')

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train", rank=trainer.global_rank)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val", rank=trainer.global_rank)
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

# ---------------------------------------------------------------------------------
class CUDACallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        # lightning update
        if int((pl.__version__).split('.')[1])>=7:
            gpu_index = trainer.strategy.root_device.index
        else:
            gpu_index = trainer.root_gpu
        torch.cuda.reset_peak_memory_stats(gpu_index)
        torch.cuda.synchronize(gpu_index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if int((pl.__version__).split('.')[1])>=7:
            gpu_index = trainer.strategy.root_device.index
        else:
            gpu_index = trainer.root_gpu
        torch.cuda.synchronize(gpu_index)
        max_memory = torch.cuda.max_memory_allocated(gpu_index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)
            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------------
# for lower lighting
class SetupCallback_low(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config, auto_resume, save_ptl_log=False):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.auto_resume = auto_resume
        self.save_ptl_log = save_ptl_log

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last_summoning.ckpt")
            trainer.save_checkpoint(ckpt_path)

        # for old version lightning
    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)
            
            if self.save_ptl_log:
                logger = logging.getLogger("pytorch_lightning")
                logger.addHandler(logging.FileHandler(os.path.join(self.logdir, "ptl_log.log")))

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            pass

# for higher lighting
class SetupCallback_high(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config, auto_resume, save_ptl_log=False):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.auto_resume = auto_resume
        self.save_ptl_log = save_ptl_log

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last_summoning.ckpt")
            trainer.save_checkpoint(ckpt_path)

    # RuntimeError: The `Callback.on_pretrain_routine_start` hook was removed in v1.8. Please use `Callback.on_fit_start` instead.
    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)
            
            if self.save_ptl_log:
                logger = logging.getLogger("pytorch_lightning")
                logger.addHandler(logging.FileHandler(os.path.join(self.logdir, "ptl_log.log")))

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))
