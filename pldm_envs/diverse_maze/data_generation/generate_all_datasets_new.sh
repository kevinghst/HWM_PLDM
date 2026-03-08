# This file generates new datasets for the diverse maze environments

# Where the datasets will be saved. CHANGE TO YOUR OWN.
OUTPUT_PATH_ROOT=/vast/wz1232

# Root path of the project. CHANGE TO YOUR OWN.
PROJECT_ROOT=/scratch/wz1232/HJEPA/environments/diverse_maze

# Generate dataset for 40maps setting.
python generate_data.py --output_path ${OUTPUT_PATH_ROOT}/40maps --config ${PROJECT_ROOT}/configs/40maps.yaml

# Generate dataset for 20maps setting. Where map layouts are subset of the 40 maps from above. 
python generate_data.py --output_path ${OUTPUT_PATH_ROOT}/20maps --config ${PROJECT_ROOT}/configs/20maps.yaml --map_path ${OUTPUT_PATH_ROOT}/40maps/train_maps.pt

# Generate dataset for 10maps setting. Where map layouts are subset of the 40 maps from above. 
python generate_data.py --output_path ${OUTPUT_PATH_ROOT}/10maps --config ${PROJECT_ROOT}/configs/10maps.yaml --map_path ${OUTPUT_PATH_ROOT}/40maps/train_maps.pt

# Generate dataset for 5maps setting. Where map layouts are subset of the 40 maps from above. 
python generate_data.py --output_path ${OUTPUT_PATH_ROOT}/5maps --config ${PROJECT_ROOT}/configs/5maps.yaml --map_path ${OUTPUT_PATH_ROOT}/40maps/train_maps.pt

# Generate test dataset for evaluation and probing. Where map layouts are disjoint from the training maps 
python generate_data.py --output_path ${OUTPUT_PATH_ROOT}/40maps_eval --config ${PROJECT_ROOT}/configs/40maps_eval.yaml --exclude_map_path ${OUTPUT_PATH_ROOT}/40maps/train_maps.pt


DATA_PATHS=(
    "${OUTPUT_PATH_ROOT}/40maps"
    "${OUTPUT_PATH_ROOT}/20maps"
    "${OUTPUT_PATH_ROOT}/10maps"
    "${OUTPUT_PATH_ROOT}/5maps"
    "${OUTPUT_PATH_ROOT}/40maps_eval"
)

# render the datasets. save images as numpy
for DATA_PATH in "${DATA_PATHS[@]}"; do
    python render_data.py --data_path "$DATA_PATH"
    python postprocess_images.py --data_path "$DATA_PATH"

    # optional: also transform the data into format compatible with ogbench (https://github.com/seohongpark/ogbench)
    # python prepare_npz.py --data_path "$DATA_PATH"
done

# Generate OOD evaluation trials for the 5 maps setting
python generate_ood_trials.py --data_path ${OUTPUT_PATH_ROOT}/5maps
