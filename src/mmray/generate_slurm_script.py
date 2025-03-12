import argparse
import os
from subprocess import Popen
from subprocess import PIPE
import random


def generate_slurm_script():
    argparser = argparse.ArgumentParser(
        "Make a slurm script for running Ray on multiple nodes"
    )
    argparser.add_argument("--job_name", type=str, default="ray")
    argparser.add_argument("--partition", type=str, default="dinglab")
    argparser.add_argument("--time", type=str, default="1:00:00")
    argparser.add_argument("--num_nodes", type=int, default=1)
    argparser.add_argument("--cpus_per_node", type=int, default=1)
    argparser.add_argument("--gpus_per_node", type=int, default=0)
    argparser.add_argument("--mem_per_node", type=str, default="10G")
    argparser.add_argument(
        "--output",
        type=str,
        default="./slurm_output/ray.out",
        help="slurm output file name. It could include the file's path",
    )
    argparser.add_argument(
        "--include_dashboard",
        type=bool,
        default=False,
        help="whether to include the dashboard",
    )
    argparser.add_argument(
        "--script_path",
        type=str,
        help="path of the script to run",
        required=True,
    )

    ## parse arguments
    args = argparser.parse_args()

    ## set directives
    directive = f"#!/bin/bash\n"
    directive += f"#SBATCH --job-name={args.job_name}\n"
    directive += f"#SBATCH --partition={args.partition}\n"
    directive += f"#SBATCH --time={args.time}\n"
    directive += f"#SBATCH --nodes={args.num_nodes}\n"
    directive += f"#SBATCH --tasks-per-node=1\n"
    directive += f"#SBATCH --cpus-per-task={args.cpus_per_node}\n"
    if args.gpus_per_node > 0:
        directive += f"#SBATCH --gres=gpu:rtx_a5000:{args.gpus_per_node}\n"
    directive += f"#SBATCH --mem={args.mem_per_node}M\n"

    directive += "#SBATCH --open-mode=truncate\n"
    if args.output[-1] == "/":
        raise ValueError("output should be a file name, not a directory")

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    directive += f"#SBATCH --output={args.output}\n"
    directive += "\n"

    ## commands for starting ray
    command = "### Commands for starting ray ###\n"
    command += "module load cuda\n"
    command += "export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE\n"
    command += "\n"

    command += "nodelist=$(scontrol show hostnames $SLURM_JOB_NODELIST)\n"
    command += 'echo "node list: $nodelist"\n'
    command += "\n"

    command += "nodes_array=($nodelist)\n"
    command += "head_node=${nodes_array[0]}\n"
    command += 'head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" sleep 1 > /dev/null 2>&1; hostname --ip-address)\n'
    command += 'echo "head node: $head_node"\n'
    command += 'echo "head node ip: $head_node_ip"\n'
    command += "\n"

    ## There are some networking caveats when using slurm with ray. See https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-networking-caveats for details.
    ## Here to prevent the port collision, we pre-assign different port ranges to different lab members.

    port = {
        "xqding": range(6000, 6100),
        "xding07": range(6000, 6100),
        "adhar01": range(7000, 7100),
    }
    ray_client_server_port = {
        "xqding": range(6200, 6300),
        "xding07": range(6200, 6300),
        "adhar01": range(7200, 7300),
    }

    node_manager_port = {
        "xqding": range(6400, 6500),
        "xding07": range(6400, 6500),
        "adhar01": range(7400, 7500),
    }
    object_manager_port = {
        "xqding": range(6600, 6700),
        "xding07": range(6600, 6700),
        "adhar01": range(7600, 7700),
    }
    runtime_env_agent_port = {
        "xqding": range(6800, 6900),
        "xding07": range(6800, 6900),
        "adhar01": range(7800, 7900),
    }

    min_worker_port = {"xqding": 10000, "xding07": 10000, "adhar01": 12000}
    max_worker_port = {"xqding": 12000, "xding07": 12000, "adhar01": 13000}

    user_name = os.environ["USER"]
    port = random.choice(port[user_name])
    ray_client_server_port = random.choice(ray_client_server_port[user_name])

    node_manager_port = random.choice(node_manager_port[user_name])
    object_manager_port = random.choice(object_manager_port[user_name])
    runtime_env_agent_port = random.choice(runtime_env_agent_port[user_name])

    min_worker_port = min_worker_port[user_name]
    max_worker_port = max_worker_port[user_name]

    command += f"port={port}\n"
    command += f"head_node_ip_with_port=$head_node_ip:$port\n"
    command += "\n"

    command += 'echo "starting head node"\n'
    command += 'srun --nodes=1 --ntasks=1 --nodelist="$head_node" \\\n'
    command += "     ray start --head --block \\\n"
    command += "     --num-cpus=$SLURM_CPUS_ON_NODE \\\n"
    if args.gpus_per_node > 0:
        command += "     --num-gpus=$SLURM_GPUS_ON_NODE \\\n"

    command += f"     --port={port} \\\n"
    command += f"     --ray-client-server-port={ray_client_server_port} \\\n"

    command += f"     --node-manager-port={node_manager_port} \\\n"
    command += f"     --object-manager-port={object_manager_port} \\\n"
    command += f"     --runtime-env-agent-port={runtime_env_agent_port} \\\n"

    command += f"     --min-worker-port={min_worker_port} \\\n"
    command += f"     --max-worker-port={max_worker_port} \\\n"

    command += f"     --include-dashboard={args.include_dashboard} &\n"
    command += "sleep 5\n"
    command += "\n"

    command += 'echo "starting worker nodes"\n'
    command += "for i in $(seq 1 $(( $SLURM_JOB_NUM_NODES-1 )) ); do\n"
    command += "    node=${nodes_array[$i]}\n"
    command += '    echo "STARTING WORKER $i at $node"\n'
    command += "    srun --nodes=1 --ntasks=1 --nodelist=$node \\\n"
    command += "         ray start --address=$head_node_ip_with_port --block \\\n"
    command += "         --num-cpus=$SLURM_CPUS_ON_NODE \\\n"
    if args.gpus_per_node > 0:
        command += "     --num-gpus=$SLURM_GPUS_ON_NODE \\\n"

    command += f"         --node-manager-port={node_manager_port} \\\n"
    command += f"         --object-manager-port={object_manager_port} \\\n"
    command += f"         --runtime-env-agent-port={runtime_env_agent_port} &\n"

    command += "done\n"
    command += "\n"

    command += "export RAY_ADDRESS=$head_node_ip_with_port\n"

    ## commands for running the script
    ## get the path of python executable
    p1 = Popen(["which", "python"], stdout=PIPE, stderr=PIPE)
    output = p1.communicate()[0].decode()
    PYTHON = output.strip()

    command += "\n"
    command += "#######################################\n"
    command += "### Commands for running the script ###\n"
    command += "#######################################\n"
    command += f"{PYTHON} -u {args.script_path}\n"

    print(directive + command)


if __name__ == "__main__":
    generate_slurm_script()
