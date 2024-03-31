"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys
import argparse
import os
from improved_ddpm.unet import UNetModel as ImprovedDDPM_Model
from ddim.diffusion import Model as DDPM_Model
import yaml
import numpy as np
import torch as th
import yaml
import torch.distributed as dist
from evaluations.th_evaluator import FIDAndIS
from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from ddim.runners.diffusion import Diffusion

def eval_fid(args, dm, model, x_shape):
    model.eval().requires_grad_(False)
    fid_is = FIDAndIS()
    fid_is.set_ref_batch(args.ref_batch)
    (
        ref_fid_stats,
        _,
        _,
    ) = fid_is.get_ref_batch(args.ref_batch)
    
    all_preds = []
    all_images = []
    while(len(all_preds) * args.batch_size < args.num_samples):
        noise = th.randn(args.batch_size, *x_shape).to(dtype=th.float32, device=dist_util.dev())
        sample, _ = dm.sample_image(noise, model)
        pred, _, _, _, _ = fid_is.get_preds(sample.clamp(-1, 1))

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_pred = [th.zeros_like(pred) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_pred, pred)  # gather not supported with NCCL
        all_preds.extend([pred.cpu().numpy() for pred in gathered_pred])
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_preds) * args.batch_size} samples")
    all_preds = np.concatenate(all_preds, axis=0)[:args.num_samples]
    fid_stats = fid_is.get_statistics(all_preds, -1)
    fid = ref_fid_stats.frechet_distance(fid_stats)
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        dir_path = os.path.join(args.exp)
        os.makedirs(dir_path, exist_ok=True)
        out_path = os.path.join(dir_path, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    logger.log(f"sampling complete, fid: {fid:.2f}")
    dist.barrier()
    return fid
        
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.gpu, args.seed)
    logger.configure(dir=args.exp)
        
    logger.log("Conducting Command: %s", " ".join(sys.argv))
    logger.log("creating model and diffusion...")
    # load diffusion model config
    with open(args.dm_config, "r") as f:
        dm_config = yaml.safe_load(f)
    dm_config = dict2namespace(dm_config)
    dm_config.deivce = dist_util.dev()
    dm = Diffusion(args, dm_config, rank=dist.get_rank())
    if args.image_size == 64:
        model = ImprovedDDPM_Model(
                in_channels=dm_config.model.in_channels,
                model_channels=dm_config.model.model_channels,
                out_channels=dm_config.model.out_channels,
                num_res_blocks=dm_config.model.num_res_blocks,
                attention_resolutions=dm_config.model.attention_resolutions,
                dropout=dm_config.model.dropout,
                channel_mult=dm_config.model.channel_mult,
                conv_resample=dm_config.model.conv_resample,
                dims=dm_config.model.dims,
                use_checkpoint=dm_config.model.use_checkpoint,
                num_heads=dm_config.model.num_heads,
                num_heads_upsample=dm_config.model.num_heads_upsample,
                use_scale_shift_norm=dm_config.model.use_scale_shift_norm
            ).to(dist_util.dev())
    elif args.image_size == 32:
        model = DDPM_Model(dm_config).to(dist_util.dev())
    else:
        raise ValueError

    logger.log(f"loading {args.model_path}")
    state = th.load(args.model_path)
    model.load_state_dict(state)
    dist_util.sync_params(model.parameters())
    dist_util.sync_params(model.buffers())
    eval_fid(args, dm, model, [3, args.image_size, args.image_size])
        
    

def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        exp="search/cifar10_1w_5w",
        gpu=None,
        dm_config=None,
        keep_samples=False,
        save_mode="image",
        enable_skip_scaler=False,
        skip_scaler_t_interval=None,
        load_weight_list="public_exp/ema_test/50_interval_save/additional_ckpts",
        data_num=5000,
        interval=1000,
        start_id=10050,
        end_id=50000,
        split="test",
        ref_batch=None,
        evo_kwargs_path="checkpoints/ema_search_kwargs.yml",
        sample_type='dpmsolver++', 
        skip_type='logSNR', 
        base_samples=None, 
        timesteps=15, 
        dpm_solver_order=3, 
        eta=0.0, 
        fixed_class=None, 
        dpm_solver_atol=0.0078, 
        dpm_solver_rtol=0.05, 
        dpm_solver_method='multistep', 
        dpm_solver_type='dpmsolver',
        scale=None,
        denoise=False,
        thresholding=False,
        lower_order_final=False, 
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == "__main__":
    main()
