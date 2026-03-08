
# Setup for NYU

## Singularity Setup
```
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-1=25GB-500K.ext3.gz .

gunzip overlay-25GB-500K.ext3.gz

singularity exec --nv --overlay overlay-25GB-500K.ext3:rw --overlay /scratch/work/public/singularity/anaconda3-2024.06-1.sqf:rw /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash
```

Continue follow [NYU Guide](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda) to finish setup. Skip the initial steps, go straight to instruction on setting up conda - *Now, inside the container, download and install miniforge to /ext3/miniforge3*...

At last, create a new conda environment for the project:

```
conda create -n hjepa python=3.9.21 -y
```

## Repo Setup

```
singularity exec --nv --overlay overlay-25GB-500K.ext3:rw --overlay /scratch/work/public/singularity/anaconda3-2024.06-1.sqf:rw /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash

source /ext3/env.sh

conda activate hjepa

git clone git@github.com:vladisai/HJEPA.git

cd HJEPA

pip install -r requirements.txt

pip install -e .
```

## Run Experiments

1. Go to `environments/`, follow instructions to set up dataset for the environment of your hoice
2. Go to `hjepa/`, follow instruction to run training or evaluation
