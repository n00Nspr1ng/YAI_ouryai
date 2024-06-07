exp=3d_diffuser_actor

tasks=(
    yai_pick_and_place
)
data_dir=./data/yaicon-ouryai/small/
num_episodes=1 #num_demos
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
use_instruction=1
max_tries=2
verbose=1
interpolation_length=2
single_task_gripper_loc_bounds=0
embedding_dim=120
cameras="left_shoulder,right_shoulder,wrist,front"
fps_subsampling_factor=5
lang_enhanced=0
relative_action=0
checkpoint=YAI_ouryai/files/yai_pick_and_place_best.pth
quaternion_format=xyzw

# Train set seed: 3 6 10 11 16 17 21 27 34 37 38 42 44 46 48 50 55 56 57 59
# Successful seed: 21 (partial), 42 (perfect)
seeds=(42)

num_seeds=${#seeds[@]}
num_ckpts=${#tasks[@]}

for ((i=0; i<$num_ckpts; i++)); do
    for ((j=0; j<$num_seeds; j++)); do
        CUDA_LAUNCH_BLOCKING=1 python YAI_ouryai/visualization/gather_data.py \
        --tasks ${tasks[$i]} \
        --checkpoint $checkpoint \
        --diffusion_timesteps 100 \
        --fps_subsampling_factor $fps_subsampling_factor \
        --lang_enhanced $lang_enhanced \
        --relative_action $relative_action \
        --num_history 3 \
        --test_model 3d_diffuser_actor \
        --cameras $cameras \
        --verbose $verbose \
        --action_dim 8 \
        --collision_checking 0 \
        --predict_trajectory 1 \
        --embedding_dim $embedding_dim \
        --rotation_parametrization "6D" \
        --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
        --data_dir $data_dir \
        --num_episodes $num_episodes \
        --output_file visual_log/$exp/seed${seeds[$j]}/${tasks[$i]} \
        --use_instruction $use_instruction \
        --instructions instructions/yaicon-ouryai/instructions.pkl \
        --variations {0..60} \
        --max_tries $max_tries \
        --max_steps 15 \
        --seed ${seeds[$j]} \
        --gripper_loc_bounds_file $gripper_loc_bounds_file \
        --gripper_loc_bounds_buffer 0.04 \
        --quaternion_format $quaternion_format \
        --interpolation_length $interpolation_length \
        --dense_interpolation 1
    done
done


#close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap place_cups