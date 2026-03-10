# Repo Setup

Tested on python 3.9, CUDA 13.0

```
git clone git@github.com:kevinghst/HWM_PLDM.git

cd HWM_PLDM

conda create -n pldm python=3.9 -y

conda activate pldm

pip install -r requirements.txt

pip install -e .
```

## MuJoCo 2.1 for d4rl + mujoco-py
mkdir -p "$HOME/.mujoco"
cd "$HOME/.mujoco"
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz

## Runtime env
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
export D4RL_SUPPRESS_IMPORT_ERROR=1

# Run Experiments

1. Go to `pldm_envs/`, follow instructions to set up dataset for the environment of your choice
2. Go to `pldm/`, follow instruction to run training or evaluation


# Datasets

To see the datasets we used to train our models, see folders inside pldm_envs/. The readmes there will guide you on how to download and set up the datasets