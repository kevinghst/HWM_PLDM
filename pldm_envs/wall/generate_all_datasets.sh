# OUTPUT_PATH=<path_to_output_folder>
OUTPUT_PATH_ROOT=/volume/data/code_release_test

mkdir -p ${OUTPUT_PATH_ROOT}

# 3M is the number used in the paper.
# Set to 1000 for quick testing

N_TRANSITIONS=1000
# N_TRANSITIONS=3000000
#
python generate_data.py --config_path configs/good_quality_data.yaml --num_transitions ${N_TRANSITIONS} --output_path ${OUTPUT_PATH_ROOT}/good_quality_data.npz
python generate_data.py --config_path configs/good_quality_data.yaml --num_transitions ${N_TRANSITIONS} --output_path ${OUTPUT_PATH_ROOT}/good_quality_data_2.npz

# varying the sequence length
for len in 17 33 65; do
    python generate_data.py --config_path configs/len=${len}.yaml --num_transitions ${N_TRANSITIONS} --output_path ${OUTPUT_PATH_ROOT}/len_${len}.npz
done


# datasets for the experiment with dataset size
# for ds_size in 634 1269 5078 20312 81250 325K 1500K; do
#    python generate_data.py --config_path configs/good_quality_data.yaml --num_transitions ${ds_size} --output_path ${OUTPUT_PATH_ROOT}/ds_size_${ds_size}.npz
# done


# Generating trajectories with random actions.
python generate_data.py --config_path configs/random_trajectories.yaml --num_transitions ${N_TRANSITIONS} --output_path ${OUTPUT_PATH_ROOT}/random_trajectories.npz


# Generate mixtures of random and non-random data
GOOD_QUALITY_DATASET_PATH=${OUTPUT_PATH_ROOT}/good_quality_data.npz
RANDOM_TRAJ_DATASET_PATH=${OUTPUT_PATH_ROOT}/random_trajectories.npz

#for fraction in 0.001 0.01 0.02 0.04 0.08 0.16; do
for fraction in 0.001; do
    OUTPUT_PATH=${OUTPUT_PATH_ROOT}/noise_mix_${fraction}.npz
    python combine_two_datasets.py ${GOOD_QUALITY_DATASET_PATH} ${RANDOM_TRAJ_DATASET_PATH} \
        --output_path ${OUTPUT_PATH} \
        --fraction ${fraction}
done


# To build lower wall crossing rate, you need at least two datasets.
# This script selects the trajectories that do not go through door
# From both datasets to bulid the resulting dataset.
DS1=${OUTPUT_PATH_ROOT}/good_quality_data.npz
DS2=${OUTPUT_PATH_ROOT}/good_quality_data_2.npz
OUTPUT_PATH=${OUTPUT_PATH_ROOT}/wc_rate_0.npz

# The desirable fraction of trajectories that go through the door.
WALL_CROSSING_RATE=0 

python select_wc.py ${DS1} ${DS2} --wc_rate ${WALL_CROSSING_RATE} --output_path ${OUTPUT_PATH}
