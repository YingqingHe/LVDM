import os, sys, datetime, glob
import logging
import argparse
from functools import partial
from packaging import version
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from pytorch_lightning.plugins import DDPPlugin

from lvdm.utils.common_utils import instantiate_from_config, str2bool
from lvdm.utils.log import set_ptl_logger

# if int((pl.__version__).split('.')[1])>=7:
#     from pytorch_lightning.strategies import DDPStrategy,DDPShardedStrategy
# else:
#     from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.plugins import DDPPlugin,DeepSpeedPlugin,DDPShardedPlugin
# from pytorch_lightning.strategies import DeepSpeedStrategy,DDPSpawnShardedStrategy
# if int((pl.__version__).split('.')[1])>=7:
#     from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default


# ---------------------------------------------------------------------------------
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?", help="postfix for logdir")
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=False, nargs="?", help="train")
    parser.add_argument("-v", "--val", type=str2bool, const=True, default=False, nargs="?", help="val")
    parser.add_argument("--test", type=str2bool, const=True, default=False, nargs="?", help="test")
    parser.add_argument("--no-test", type=str2bool, const=True, default=False, nargs="?", help="disable test")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-d", "--debug", type=str2bool, nargs="?", const=True, default=False, help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=True, help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument("--increase_log_steps", type=str2bool, nargs="?", const=True, default=True, help="")
    parser.add_argument("--auto_resume", type=str2bool, nargs="?", const=False, default=False, help="")
    parser.add_argument("--load_from_checkpoint", type=str, default="", help="")
    return parser

# ---------------------------------------------------------------------------------
def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

# ---------------------------------------------------------------------------------
class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ---------------------------------------------------------------------------------
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, 
                 shuffle_test_loader=False, shuffle_val_dataloader=False, 
                 use_worker_init_fn=False,
                 test_max_n_samples=None, val_max_n_samples=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap
        self.test_max_n_samples = test_max_n_samples
        self.val_max_n_samples = val_max_n_samples

    def prepare_data(self):
        pass

    def setup(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        loader = DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          worker_init_fn=None, collate_fn=None,
                          )
        return loader

    def _val_dataloader(self, shuffle=False):
        if self.val_max_n_samples is not None:
            dataset = torch.utils.data.Subset(self.datasets["validation"], list(range(self.val_max_n_samples)))
        else:
            dataset = self.datasets["validation"]
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=None,
                          shuffle=shuffle, 
                          collate_fn=None,
                          )

    def _test_dataloader(self, shuffle=False):
        if self.test_max_n_samples is not None:
            dataset = torch.utils.data.Subset(self.datasets["test"], list(range(self.test_max_n_samples)))
        else:
            dataset = self.datasets["test"]
        return DataLoader(dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=None, shuffle=shuffle,
                          collate_fn=None,
                          )

    def _predict_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=None,
                          collate_fn=None,
                          )

# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())
    
    # make dir name: (now time) + name + postfix
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if opt.auto_resume:
        # no time
        nowname = opt.name + opt.postfix
    else:
        nowname = now + "_" + opt.name + opt.postfix
    logdir = os.path.join(opt.logdir, nowname)
    
    if opt.auto_resume:
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        if os.path.exists(ckpt):
            resume = True
            try:
                tmp = torch.load(ckpt, map_location='cpu')
                e = tmp['epoch']
                gs = tmp['global_step']
                print(f"[INFO] Resume from epoch {e}, global step {gs}!")
                del tmp
            except:
                try:
                    print("Load last.ckpt failed!")
                    ckpts = sorted([f for f in os.listdir(os.path.join(logdir, "checkpoints")) if not os.path.isdir(f)])
                    print(f"all avaible checkpoints: {ckpts}")
                    ckpts.remove("last.ckpt")
                    if "trainstep_checkpoints" in ckpts:
                        ckpts.remove("trainstep_checkpoints")
                    ckpt_path = ckpts[-1]
                    ckpt = os.path.join(logdir, "checkpoints", ckpt_path)
                    print(f"Select resuming ckpt: {ckpt}")
                except ValueError:
                    print("Load last.ckpt failed! and there is no other ckpts")
                    
            opt.resume_from_checkpoint = ckpt
            print(f"[INFO] resume from: {ckpt}")
        else:
            resume = False
            opt.resume_from_checkpoint = None
            print(f"[INFO] no checkpoint found in current logdir: {os.path.join(logdir, 'checkpoints')}")
    else:
        resume = False
    
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    os.makedirs(logdir, exist_ok=True)
    print('logdir: ', logdir)

    if opt.test:
        set_ptl_logger(logdir, 'test')
    else:
        set_ptl_logger(logdir, 'train')

    # disable transformer warning
    logging.set_verbosity_error()

    seed_everything(opt.seed)
    
    # ---------------------------------------------------------------------------------

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        
        # default to ddp
        if "accelerator" not in trainer_config:
            # lightining update
            if int((pl.__version__).split('.')[1])>=7:
                trainer_config["accelerator"] = "cuda"
            else:
                trainer_config["accelerator"] = "ddp"
            print('Set DDP mode')

        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False

        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        config.model['ckptdir'] = ckptdir
        config.model.params['logdir'] = logdir
        model = instantiate_from_config(config.model)
        
        # ckpt
        if opt.load_from_checkpoint:
            config.model.load_from_checkpoint = opt.load_from_checkpoint
        if "load_from_checkpoint" in config.model and config.model.load_from_checkpoint and not resume:
            try:
                model = model.load_from_checkpoint(config.model.load_from_checkpoint, **config.model.params)
            except:
                # avoid size mismatch
                # gpu_id = opt.gpus.split(",")[0]
                # state_dict = torch.load(config.model.load_from_checkpoint, map_location=f"cuda:{gpu_id}")['state_dict']
                state_dict = torch.load(config.model.load_from_checkpoint, map_location=f"cpu")['state_dict']
                model_state_dict = model.state_dict()
                for n, p in model_state_dict.items():
                    if p.shape != state_dict[n].shape:
                        print(f"Skip load parameter [{n}] from pretrained! ")
                        state_dict.pop(n)
                model_state_dict.update(state_dict)
                model.load_state_dict(model_state_dict)

        # trainer and callbacks
        trainer_kwargs = dict()

        # make logger
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                # https://github.com/Lightning-AI/lightning/issues/13958
                # The test-tube package is no longer maintained and PyTorch Lightning will remove the :class:´TestTubeLogger´ in v1.7.0.
                "target": "pytorch_lightning.loggers.CSVLogger" if int((pl.__version__).split('.')[1])>=7 else "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        print("increase_log_steps: ", opt.increase_log_steps)
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "lvdm.utils.callbacks.SetupCallback_high" if int((pl.__version__).split('.')[1])>=7 else "lvdm.utils.callbacks.SetupCallback_low" ,
                "params": {
                    "resume": '',
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "auto_resume": opt.auto_resume,
                }
            },
            "image_logger": {
                "target": "lvdm.utils.callbacks.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True,
                    "increase_log_steps": opt.increase_log_steps
                }
            },
            "learning_rate_logger": {
                "target": "lvdm.utils.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                }
            },
            "cuda_callback": {
                "target": "lvdm.utils.callbacks.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        # default strategy config
        default_strategy_dict = {
            "target": "pytorch_lightning.strategies.DDPShardedStrategy"
        }

        if "strategy" in lightning_config:
            strategy_cfg = lightning_config.strategy
        else:
            strategy_cfg = OmegaConf.create()
            strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)

        if int((pl.__version__).split('.')[1])>=7:
            trainer_kwargs['precision'] = lightning_config.get('precision', 32)
            print(f'set precision={trainer_kwargs["precision"]}')
            print('lightning_config',lightning_config)
            # strategy can be str
            if type(strategy_cfg) == str:
                trainer_kwargs["strategy"] = strategy_cfg
            else:
                # default strategy is ddp shared
                trainer_kwargs["strategy"] = instantiate_from_config(strategy_cfg)
            print(f'strategy')
            print(trainer_kwargs["strategy"])
        else:
            print('low version ptl, no ddp shared')
            find_unused_parameters=lightning_config.get("find_unused_parameters", False)
            trainer_kwargs["plugins"] = DDPPlugin(find_unused_parameters=find_unused_parameters)

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir

        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1

        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
            # adjust the log batch freq to the actual forward steps (not the optimize step)
            lightning_config.callbacks.image_logger.params.batch_frequency = lightning_config.callbacks.image_logger.params.batch_frequency / accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches

        # data
        if getattr(config.data, 'auto_cal_bs', False):
            bs_per_gpu = config.data.params.batch_size * accumulate_grad_batches
            total_bs = ngpu * lightning_config.trainer.num_nodes \
                       * bs_per_gpu
            print(f'Actual total batch size = {total_bs}')
            config.data.params.train.params['bs_per_gpu'] = bs_per_gpu
            if "validation" in config.data.params:
                config.data.params.validation.params['bs_per_gpu'] = bs_per_gpu
        
        data = instantiate_from_config(config.data)
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        scale_lr = opt.scale_lr and getattr(config.model, 'scale_lr', True)
        if scale_lr:
            num_nodes = lightning_config.trainer.num_nodes
            model.learning_rate = ngpu * num_nodes * bs * base_lr * accumulate_grad_batches
            print("Setting learning rate to {:.2e} = {} (num_gpus) * {} (num_nodes) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, ngpu, num_nodes, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last_summoning.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if opt.val:
            trainer.validate(model, data)
        if opt.test or (not opt.no_test and not trainer.interrupted):
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
    # ---------------------------------------------------------------------------------
