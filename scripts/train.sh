####################################################################
# Training EDM models on class-conditional ImageNet-64, and LSUN 256
####################################################################

mpiexec -n 8 python edm_train.py --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 4096 --image_size 64 --lr 0.0001 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler lognormal --use_fp16 True --weight_decay 0.0 --weight_schedule karras --data_dir /path/to/imagenet

python -m orc.diffusion.scripts.train_imagenet_edm --attention_resolutions 32,16,8 --class_cond False --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 256 --image_size 256 --lr 0.0001 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 True --use_scale_shift_norm False --weight_decay 0.0 --weight_schedule karras --data_dir /path/to/lsun_bedroom

#########################################################################
# Consistency distillation on CIFAR10, class-conditional ImageNet-64, and LSUN 256
#########################################################################

## L_CD^N on CIFAR-10
Coming soon

## L_CD^N (l2) on ImageNet-64
mpiexec -n 8 python cm_train.py --training_mode consistency_distillation --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 --loss_norm l2 --lr_anneal_steps 0 --teacher_model_path /path/to/edm_imagenet64_ema.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 2048 --image_size 64 --lr 0.000008 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/data

## L_CD^N (LPIPS) on ImageNet-64
mpiexec -n 8 python cm_train.py --training_mode consistency_distillation --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path /path/to/edm_imagenet64_ema.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 2048 --image_size 64 --lr 0.000008 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/data

## L_CD^N (l2) on LSUN 256
mpiexec -n 8 python cm_train.py --training_mode consistency_distillation --sigma_max 80 --sigma_min 0.002 --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 --loss_norm l2 --lr_anneal_steps 0 --teacher_model_path /path/to/edm_bedroom256_ema.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.9999,0.99994,0.9999432189950708 --global_batch_size 256 --image_size 256 --lr 0.00001 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/bedroom256

## L_CD^N (LPIPS) on LSUN 256
mpiexec -n 8 python cm_train.py --training_mode consistency_distillation --sigma_max 80 --sigma_min 0.002 --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path /path/to/edm_bedroom256_ema.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.9999,0.99994,0.9999432189950708 --global_batch_size 256 --image_size 256 --lr 0.00001 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/bedroom256

#########################################################################
# Consistency training on CIFAR10, class-conditional ImageNet-64, and LSUN 256
#########################################################################

## L_CT^N on CIFAR-10
Coming soon

## L_CT^N on ImageNet-64
mpiexec -n 8 python cm_train.py --training_mode consistency_training --target_ema_mode adaptive --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 200 --total_training_steps 800000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path /path/to/edm_imagenet64_ema.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 2048 --image_size 64 --lr 0.0001 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/imagenet64

## L_CT^N on LSUN 256
mpiexec -n 8 python cm_train.py --training_mode consistency_training --target_ema_mode adaptive --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 150 --total_training_steps 1000000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path /path/to/edm_bedroom256_ema.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.9999,0.99994,0.9999432189950708 --global_batch_size 256 --image_size 256 --lr 0.00005 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/bedroom256
