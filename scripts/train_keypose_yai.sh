main_dir=yaicon_ouryai #Actor_18Peract_100Demo_multitask

dataset=data/yaicon_ouryai/packaged/train
valset=data/yaicon_ouryai/packaged/val

lr=1e-4
dense_interpolation=1
interpolation_length=2
num_history=3
diffusion_timesteps=100
B=4
C=120
ngpus=1
quaternion_format=xyzw

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_trajectory.py \
    --tasks yai_pick_and_place \
    --dataset $dataset \
    --valset $valset \
    --instructions instructions/yaicon_ouryai/instructions.pkl \
    --gripper_loc_bounds tasks/18_peract_tasks_location_bounds.json \
    --num_workers 1 \
    --train_iters 250000 \
    --embedding_dim $C \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --val_freq 5000 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 4 \
    --cache_size 600 \
    --cache_size_val 0 \
    --keypose_only 1 \
    --variations {0..1} \
    --lr $lr\
    --num_history $num_history \
    --cameras wrist right_shoulder left_shoulder front\
    --max_episodes_per_task -1 \
    --quaternion_format $quaternion_format \
    --run_log_dir C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps