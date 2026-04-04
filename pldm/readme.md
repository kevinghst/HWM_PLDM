## Diverse Mazes

First, download the pretrained world model weights by running 
```
python download_ckpt_from_hf.py --out-dir <repo_root>/pldm/pretrained
```

Which will download two ckpts:

* PLDM (1 level) ckpt: `3-9-1-seed248_epoch=3_sample_step=15465472.ckpt`
* HWM (2 levels) ckpt: `load_from_l1248-seed248_epoch=5_sample_step=10789632.ckpt`

To evaluate hierarchical planning on the downloaded HWM ckpt:

```
python train.py --config configs/diverse_maze/icml/large_diverse_25maps_l2.yaml --values eval_only=true load_l1_only=false load_checkpoint_path=<repo_root>/pldm/pretrained/load_from_l1248-seed248_epoch=5_sample_step=10789632.ckpt
```

To evaluate flat planning on the downloaded PLDM ckpt:

```
python train.py --config configs/diverse_maze/icml/large_diverse_25maps.yaml --values eval_only=true load_checkpoint_path=<repo_root>/pldm/pretrained/3-9-1-seed248_epoch=3_sample_step=15465472.ckpt
```

To train the HWM (2 levels) on the large-maze setting by loading the downloaded level 1 PLDM model , run:

```
python train.py --config configs/diverse_maze/icml/large_diverse_25maps_l2.yaml
```

If you prefer to train the level-1 PLDM world model from scratch, run:

```
python train.py --config configs/diverse_maze/icml/large_diverse_25maps.yaml
```

Then later if you want to train a HWM model by loading the newly trained level 1 WM, update `load_checkpoint_path` in `configs/diverse_maze/icml/large_diverse_25maps_l2.yaml` to point to your trained level-1 checkpoint, and run the level-2 training command above.

## Hyperparameter tuning

Hyperparameters ($\alpha, \beta, \lambda, \delta, \omega$) should be tuned for any new environment.

Within a given environment, hyperparameters should be tuned for different offline datasets that have significant differences in data distributions.

To reduce the hyperparamter search space for a given setting, one idea is to take the hyperparameters for the closest setting, get lower and upper bounds for each parameter by dividing and multiplying it by a factor (eg: 3) respectively, and perform a random search within the lower and upper bounds of all parameters.
