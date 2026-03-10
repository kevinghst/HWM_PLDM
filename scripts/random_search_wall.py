import argparse
import re
import random

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--seed_start", default=1, type=int)
parser.add_argument("--seeds", default=1, type=int)
parser.add_argument("--sbatch", action="store_true")
parser.add_argument("--output_dir", type=str)
parser.add_argument("--run_name", type=str)
parser.add_argument("--quick_debug", action="store_true")
parser.add_argument("--other_dir", action="store_true")
parser.add_argument("--amd", action="store_true")
parser.add_argument("--bash_script", type=str, default="run_4.sbatch")

# random search trials
parser.add_argument("--trials", default=60, type=int)

args = parser.parse_args()

if not args.quick_debug and "dev" not in args.config:
    assert args.output_dir is not None
    assert args.run_name is not None

# core command
command = [
    "python",
    "hjepa/train.py",
    f"--configs hjepa/configs/{args.config}",
    "--values",
    f"output_dir={args.output_dir}",
    f"run_name={args.run_name}",
]

custom_command = []

discrete_param_grid = {}
var_mappings = {}

command += custom_command

custom_command = []

command += custom_command


def get_next_name(s):
    components = s.split("-")
    x, y, a, z_str = components
    if z_str[-1] == "a":
        l_str = f"{int(z_str[:-1]) + 1}a"
    else:
        l_str = str(int(z_str) + 1)
    new_s = f"{x}-{y}-{a}-{l_str}"
    return new_s


def get_command():
    sub_command = []
    command_str = " ".join(command + sub_command)
    return command_str


param_grid = {
    "base_lr": [0.0026, 0.023],
    "objectives_l1.idm.coeff": [1.13, 10.23],
    "objectives_l1.vicreg.std_coeff": [2.47, 22.2],
    "objectives_l1.vicreg.std_coeff_t": [0.087, 0.78],
    "objectives_l1.vicreg.cov_coeff": [0.82, 7.4],
}


all_combinations = []
for i in range(args.trials):
    params = []
    for key, param_range in param_grid.items():
        lower, upper = param_range
        sample = random.uniform(lower, upper)
        sample = float("{:.5g}".format(sample))
        params.append(sample)

    for key, choices in discrete_param_grid.items():
        sample = random.choice(choices)
        params.append(sample)

    all_combinations.append(tuple(params))


output_dir = args.output_dir
run_name = args.run_name

for idx, combination in enumerate(all_combinations):
    all_keys = list(param_grid.keys()) + list(discrete_param_grid.keys())
    current_params = dict(zip(all_keys, combination))

    current_params["run_name"] = run_name
    current_params["output_dir"] = output_dir

    print(current_params)

    command_str = get_command()

    for key, val in current_params.items():
        if f" {key}=" in command_str:
            command_str = re.sub(rf" {key}=\S+", f" {key}={val}", command_str)
        else:
            command_str += f" {key}={val}"

        if key == "n_steps":
            command_str += f" val_n_steps={val}"
            command_str += f" data.d4rl_config.sample_length={val}"
            command_str += f" hjepa.l1_n_steps={val}"
            command_str += f" eval_cfg.probing.l1_depth={val}"

        if key in var_mappings:
            mapped_vars = var_mappings[key][val]
            for mv in mapped_vars:
                command_str += f" {mv}"

    print(command_str)

    bash_script = args.bash_script

    print(bash_script)
    # bashCmd = ["sbatch", f"/scratch/wz1232/HJEPA/scripts/{bash_script}"] + [command_str]
    # process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
    # output, error = process.communicate()
    # time.sleep(2)

    output_dir = get_next_name(output_dir)
    run_name = get_next_name(run_name)

    if idx == 0:
        first_command_str = command_str

# print final command str for test
test_dir = "-".join(args.output_dir.split("-")[:-1]) + "-test"
test_command_str = first_command_str.replace(args.output_dir, test_dir)

test_command_str += " wandb=false quick_debug=true"

print(test_command_str)
