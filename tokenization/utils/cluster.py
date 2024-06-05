import os
import sys
import stat
import subprocess
from pathlib import Path

CONDOR_ROOT = '/is/cluster/fast/sdwivedi/condor/tokenhmr_condor'
GPUS = {
    'v100-p16': ('\"Tesla V100-PCIE-16GB\"', 'tesla', 16000),
    'v100-p32': ('\"Tesla V100-PCIE-32GB\"', 'tesla', 32000),
    'v100-s32': ('\"Tesla V100-SXM2-32GB\"', 'tesla', 32000),
    'quadro6000': ('\"Quadro RTX 6000\"', 'quadro', 24000),
    # 'a10080GB': ('\"NVIDIA A100-SXM4-80GB\"', 'a100', 80000),
    # 'a10040GB': ('\"NVIDIA A100-SXM4-40GB\"', 'a100', 40000),
}

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def get_gpus(min_mem=10000, arch=('tesla', 'quadro', 'a100')):
    gpu_names = []
    for k, (gpu_name, gpu_arch, gpu_mem) in GPUS.items():
        if gpu_mem >= min_mem and gpu_arch in arch:
            gpu_names.append(gpu_name)

    assert len(gpu_names) > 0, 'Suitable GPU model could not be found'

    return gpu_names

def execute_task_on_cluster(
        script,
        exp_name,
        cfg_file,
        num_exp=1,
        bid_amount=300,
        num_workers=8,
        memory=64000,
        exclude_nodes='',
        gpu_min_mem=10000,
        num_gpus=1,
):

    gpus = get_gpus(min_mem=gpu_min_mem)

    gpus = ' || '.join([f'CUDADeviceName=={x}' for x in gpus])
    ROOT = get_project_root()

    os.makedirs(os.path.join(f'{CONDOR_ROOT}', exp_name), exist_ok=True)
    submission = f'executable = {ROOT}/scripts/cluster/{exp_name}_run.sh\n' \
                 'arguments = $(Process)\n' \
                 f'error = {CONDOR_ROOT}/{exp_name}/$(Cluster).$(Process).err\n' \
                 f'output = {CONDOR_ROOT}/{exp_name}/$(Cluster).$(Process).out\n' \
                 f'log = {CONDOR_ROOT}/{exp_name}/$(Cluster).$(Process).log\n' \
                 f'request_memory = {memory}\n' \
                 f'request_cpus={int(num_workers)}\n' \
                 f'request_gpus={num_gpus}\n' \
                 f'requirements={gpus}\n' \
                 f'+MaxRunningPrice = 100\n' \
                 f'+RunningPriceExceededAction = \"kill\"\n' \
                 f'queue {num_exp}\n'
    if exclude_nodes:
        ex_nodes = exclude_nodes.split('-')
        for node in ex_nodes:
            submission += f'requirements = UtsnameNodename =!= \"{node}\"\n'

    print(submission)

    with open(f'{ROOT}/scripts/cluster/{exp_name}_submit.sub', 'w') as f:
        f.write(submission)

    print(f'The logs for this experiments can be found under: condor_logs/{exp_name}')
    bash = 'export PATH=$PATH\n' \
           f'{sys.executable} {script} --cfg {ROOT}/{cfg_file} --cfg_id $1'

    with open(f'{ROOT}/scripts/cluster/{exp_name}_run.sh', 'w') as f:
        f.write(bash)

    os.chmod(f'{ROOT}/scripts/cluster/{exp_name}_run.sh', stat.S_IRWXU)

    cmd = ['condor_submit_bid', f'{bid_amount}', f'{ROOT}/scripts/cluster/{exp_name}_submit.sub']
    print('Executing ' + ' '.join(cmd))
    subprocess.call(cmd)