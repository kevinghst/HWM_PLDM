# Using the environment

The environment adheres to the openai gym interface. To use the environment, you can use the example code in `test.py`. That file also provides an example
for creating a dynamically generated dataset, and loading the offline data.

# Loading existing data

The datasets we used in our experiments are stored on google drive, and can be downloaded 
using `download_all.sh` script in `presaved_datasets` folder. Just do `bash presaved_datasets/download_all.sh` to download all the datasets into that folder, or change to a folder you prefer instead.
To save space, we did not include the image data in the datasets. To use the
datasets, you will need to render the states first.
To render the states, you can use the command below:

```bash
python render_images.py --input_path presaved_datasets/good_quality_data_no_images.npz --output_path presaved_datasets/good_quality_data.npz --config configs/good_quality_data.yaml
```

You can run this for each of the datasets. You do not need to change the config for each of them, as the only thing that is used from the config is the layout of the two rooms environment, and that is kept constant in all experiments.

To render all the datsets, run `bash presaved_datasets/render_all.sh`.

# Generating your own data

If you'd like to generate your own data, follow the example in `generate_all_datasets.sh`. Set paths and number of trajectories in that bash script, then run it to generate the datasets.
