#################################################################################
# Sampling from consistency models on CIFAR-10, and class-conditional ImageNet-64
#################################################################################

## CIFAR-10
python image_sample.py --batch_size 512 --training_mode consistency_distillation --sampler onestep --steps 18 --model_path /path/to/checkpoint --class_cond False  --use_fp16 False --weight_schedule uniform --num_samples 50000 --image_size 32 --exp ../outputs/cifar10 --timestep_type edm --model_type edm --edm_model_kwargs_path ../configs/edm-cifar10-32x32-uncond-kwargs.yml --ref_batch /path/to/your/reference/batch --gpu 0

## ImageNet-64
python image_sample.py --batch_size 512 --training_mode consistency_distillation --sampler onestep --steps 40 --model_path /path/to/checkpoint --class_cond False --weight_schedule uniform --num_samples 50000 --exp ../outputs/imagenet64 --attention_resolutions 32,16,8 --class_cond True --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --image_size 64 --ref_batch /path/to/your/reference/batch --gpu 0

###################################################################
# Multistep sampling on CIFAR-10, and class-conditional ImageNet-64 (not recommended for models enhanced by LCSC)
###################################################################

## Two-step sampling on CIFAR-10
python image_sample.py --batch_size 512 --training_mode consistency_distillation --sampler multistep --steps 18 --ts 0,10,17 --model_path /path/to/checkpoint --class_cond False  --use_fp16 False --weight_schedule uniform --num_samples 50000 --image_size 32 --exp ../outputs/cifar10 --timestep_type edm --model_type edm --edm_model_kwargs_path ../configs/edm-cifar10-32x32-uncond-kwargs.yml --ref_batch /path/to/your/reference/batch --gpu 0

## Two-step sampling on ImageNet-64
python image_sample.py --batch_size 512 --training_mode consistency_distillation --sampler multistep --steps 40 --ts 0,18,39 --model_path /path/to/checkpoint --class_cond False --weight_schedule uniform --num_samples 50000 --exp ../outputs/imagenet64 --attention_resolutions 32,16,8 --class_cond True --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --image_size 64 --ref_batch /path/to/your/reference/batch --gpu 0

###################################################################
# Sampling from DDPM on CIFAR-10, and Improved DDPM on ImageNet-64
###################################################################

## CIFAR-10
python image_sample_dm.py --image_size=32 --exp=../outputs/cifar10 --seed=0 --num_samples=50000 --model_path=/path/to/checkpoint --dm_config=../configs/ddpm-cifar10-uncond-kwargs --batch_size=1000 --ref_batch=/path/to/reference

## ImageNet-64
python image_sample_dm.py --image_size=64 --exp=../outputs/imagenet64 --seed=0 --num_samples=50000 --model_path=/path/to/checkpoint --dm_config=../configs/iddpm-imagenet64-uncond-kwargs --batch_size=500 --ref_batch=/path/to/reference

