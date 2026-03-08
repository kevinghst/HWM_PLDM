# change if you need to
OUTPUT_PATH=./presaved_datasets

FILE_ID=1NwR-ui-akIgR2xcoJHYiHjOk9U5bYCa0
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${OUTPUT_PATH}/wall_data.tar.gz"

cd $OUTPUT_PATH

tar -xvf wall_data.tar.gz

rm wall_data.tar.gz
