DATA_PATH=presaved_datasets

mkdir -p $DATA_PATH/rendered

for path in $DATA_PATH/*.npz; do
    echo "Rendering $path"
    output_path=$DATA_PATH/rendered/$(basename $path)
    # remove no_images from the name
    output_path=${output_path//_no_images/}
    echo "running rendering for $path, saving to $output_path"
    python render_images.py --input_path $path --output_path ${output_path} --config configs/good_quality_data.yaml
done
