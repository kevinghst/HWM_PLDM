# Generate data

### Generate the data from the paper

Run the following script to download the proprioceptive data, and render the observations into top-down images:

```
bash data_generation/generate_all_datasets_og.sh
```

Note that rendering code can take a while. One can split the work across multiple CPUs by using the `--workers_num` and `worker_id` flags in `data_generation/render_data.py`

The data will be saved `datasets/` by default.

### Generate new data

The following scripts will generate datasets for the large diverse maps setting. It will also create a set of OOD evaluation trials on unseen maps.

```
bash data_generation/generate_all_datasets_new.sh
```

