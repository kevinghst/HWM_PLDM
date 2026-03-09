# This file generates new datasets for the large diverse maze environments

# Root path of the project. CHANGE TO YOUR OWN.
PROJECT_ROOT=/scratch/wz1232/HWM_PLDM

# Generate dataset for 25maps setting.
python data_generation/generate_data.py --output_path ${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps --config ${PROJECT_ROOT}/pldm_envs/diverse_maze/configs/maze2d_large/25maps.yaml

# Generate dataset for OOD evaluation (probe) setting.
python data_generation/generate_data.py --output_path ${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe/ --config ${PROJECT_ROOT}/pldm_envs/diverse_maze/configs/maze2d_large/probe.yaml --exclude_map_path ${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/train_maps.pt

DATA_PATHS=(
    "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps"
    "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe"
)

# render the datasets. save images as numpy
for DATA_PATH in "${DATA_PATHS[@]}"; do
    python "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/render_data.py" --data_path "$DATA_PATH"
    python "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/postprocess_images.py" --data_path "$DATA_PATH"
done

# Generate OOD evaluation trials for the 5 maps setting
python "${PROJECT_ROOT}/pldm_envs/diverse_maze/evaluation/generate_starts_targets.py" --data_path ${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe
