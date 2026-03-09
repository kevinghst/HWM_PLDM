# This file downloads the original datasets and render them

# Root path of the project. CHANGE TO YOUR OWN.
PROJECT_ROOT=/scratch/wz1232/HWM_PLDM

# Download datasets from HF into pldm_envs/diverse_maze/datasets.
python "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/download_ds_from_hf.py" \
    --out-dir "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets"

DATA_PATHS=(
    "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps"
    "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe"
)

# render the datasets. save images as numpy
for DATA_PATH in "${DATA_PATHS[@]}"; do
    python "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/render_data.py" --data_path "$DATA_PATH"
    python "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/postprocess_images.py" --data_path "$DATA_PATH"
done
