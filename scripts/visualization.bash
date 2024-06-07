type="denoising" # "dataset" or "denoising"
file_path="visual_log/3d_diffuser_actor/seed42"
# 'data/yaicon-ouryai/packaged/train/yai_pick_and_place+0/ep12.dat'
# "visual_log/3d_diffuser_actor/seed42"

python YAI_ouryai/visualization/visualizer.py \
--type $type \
--file_path $file_path \
