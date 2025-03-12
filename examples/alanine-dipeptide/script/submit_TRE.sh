#!/bin/bash
#SBATCH --job-name=ray
#SBATCH --partition=preempt
#SBATCH --time=5:00:00
#SBATCH --nodes=4
#SBATCH --tasks-per-node=2
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/ray.out

### Commands for starting ray ###
module load cuda
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

nodelist=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo "node list: $nodelist"

nodes_array=($nodelist)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" sleep 1 > /dev/null 2>&1; hostname --ip-address)
echo "head node: $head_node"
echo "head node ip: $head_node_ip"

port=6015
head_node_ip_with_port=$head_node_ip:$port

echo "starting head node"
srun --nodes=1 --ntasks=1 --nodelist="$head_node" \
     uv run ray start --head --block \
     --num-cpus=$SLURM_CPUS_ON_NODE \
     --port=6015 \
     --ray-client-server-port=6200 \
     --node-manager-port=6411 \
     --object-manager-port=6619 \
     --runtime-env-agent-port=6882 \
     --min-worker-port=10000 \
     --max-worker-port=12000 \
     --include-dashboard=False &
sleep 5

echo "starting worker nodes"
for i in $(seq 1 $(( $SLURM_JOB_NUM_NODES-1 )) ); do
    node=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node"
    srun --nodes=1 --ntasks=1 --nodelist=$node \
         uv run ray start --address=$head_node_ip_with_port --block \
         --num-cpus=$SLURM_CPUS_ON_NODE \
         --node-manager-port=6411 \
         --object-manager-port=6619 \
         --runtime-env-agent-port=6882 &
done

export RAY_ADDRESS=$head_node_ip_with_port

#######################################
### Commands for running the script ###
#######################################
uv run python ./script/run_TRE.py

