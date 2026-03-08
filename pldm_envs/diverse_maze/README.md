# Generate data

### Generate the data from the paper

Run the following script to download the proprioceptive data, and render the observations into top-down images:

```
bash data_generation/generate_all_datasets_og.sh
```

The data will be saved `presaved_datasets/` by default.

### Generate new data

The following scripts will generate datasets for the 40maps, 20maps, 10maps, and 5maps settings. It will also create a set of OOD evaluation trials for the 5maps setting.

```
bash data_generation/generate_all_datasets_new.sh
```

