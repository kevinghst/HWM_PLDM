# This file downloads the original datasets and render them

# Where the datasets will be saved. CHANGE TO YOUR OWN.
OUTPUT_PATH_ROOT=/vast/wz1232

# Root path of the project. CHANGE TO YOUR OWN.
PROJECT_ROOT=/scratch/wz1232/HJEPA/environments/diverse_maze

# download for 40maps
gdown "https://drive.google.com/uc?id=1KIb-d-aUODYBA8ujsPJjiCPWbuB8Ny85" -O "${PROJECT_ROOT}/presaved_datasets/40maps/data.p"

# download for 20maps
gdown "https://drive.google.com/uc?id=14XZcmJUvAoccx0ig---bq2tZvvrGklSh" -O "${PROJECT_ROOT}/presaved_datasets/20maps/data.p"

# download for 10maps
gdown "https://drive.google.com/uc?id=15bMl5jnCM6LjHl-5cZL1xUL1jGjdzcBj" -O "${PROJECT_ROOT}/presaved_datasets/10maps/data.p"

# download for 5maps
gdown "https://drive.google.com/uc?id=1PjL-5wgBrktebHlV3IwPDQdOeKSXqhvK" -O "${PROJECT_ROOT}/presaved_datasets/5maps/data.p"

# download for 40maps_eval
gdown "https://drive.google.com/uc?id=1QBSVMEdy0q71zyslQ-9QsikFul1y3pR_" -O "${PROJECT_ROOT}/presaved_datasets/40maps_eval/data.p"

DATA_PATHS=(
    "${PROJECT_ROOT}/presaved_datasets/40maps"
    "${PROJECT_ROOT}/presaved_datasets/20maps"
    "${PROJECT_ROOT}/presaved_datasets/10maps"
    "${PROJECT_ROOT}/presaved_datasets/5maps"
    "${PROJECT_ROOT}/presaved_datasets/40maps_eval"
)

# render the datasets. save images as numpy
for DATA_PATH in "${DATA_PATHS[@]}"; do
    python render_data.py --data_path "$DATA_PATH"
    python postprocess_images.py --data_path "$DATA_PATH"

    # optional: also transform the data into format compatible with ogbench (https://github.com/seohongpark/ogbench)
    # python prepare_npz.py --data_path "$DATA_PATH"
done
