 #!/bin/bash

#SBATCH -J pengfei_ppo_offline
#SBATCH -N 8
#SBATCH --gpus-per-node=8
#SBATCH -t 0-4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=176
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --overcommit

set -x -e

my_hostname=$(hostname)

if [[ $my_hostname == ib-vm-* ]]; then
        # Crusoe cluster-specific configurations
        export NCCL_DEBUG=INFO
        export NCCL_IB_AR_THRESHOLD=0
        export NCCL_IB_PCI_RELAXED_ORDERING=1
        export NCCL_IB_SPLIT_DATA_ON_QPS=0
        export NCCL_IB_QPS_PER_CONNECTION=2
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export coll_hcoll_enable=0
        export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1
        export ENROOT_RESTRICT_DEV=y
        export NCCL_IGNORE_CPU_AFFINITY=0
        export NCCL_SOCKET_NTHREADS=4
        export NCCL_NSOCKS_PERTHREAD=8
        export OMPI_MCA_btl=^openib

        export TORCH_NCCL_AVOID_RECORD_STREAMS=1
        export NCCL_NVLS_ENABLE=1
        export NCCL_IB_CUDA_SUPPORT=1

        export MLFLOW_TRACKING_URI=http://172.27.28.253:8677
        export RUN_ENV="crusoe"
else

        # Data Center network configurations
        export NCCL_DEBUG=INFO
        export NCCL_IB_AR_THRESHOLD=0
        export NCCL_IB_PCI_RELAXED_ORDERING=1
        export NCCL_IB_SPLIT_DATA_ON_QPS=0
        export NCCL_IB_QPS_PER_CONNECTION=2
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export coll_hcoll_enable=0
        export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
        export ENROOT_RESTRICT_DEV=y
        export NCCL_IGNORE_CPU_AFFINITY=0
        export NCCL_SOCKET_NTHREADS=4
        export NCCL_NSOCKS_PERTHREAD=8
        export OMPI_MCA_btl=^openib
        export NCCL_IB_TIMEOUT=22 # We are seeing port error in the data-center. Thus set a large timeout as suggested by https://discuss.pytorch.org/t/socket-timeout-for-distributed-training/142471

        # Currently the mlflow server is running on a100-16. Will migrate to k8s once we stablize that.
        export MLFLOW_TRACKING_URI=http://mlflow.canada.boson.ai:5000
        export RUN_ENV="datacenter"

        export TORCH_NCCL_AVOID_RECORD_STREAMS=1
        export NCCL_NVLS_ENABLE=1
        export NCCL_IB_CUDA_SUPPORT=1
fi

OPENRLHF_PATH=/fsx/workspace/pengfei/work/OpenRLHF
IMAGE_NAME="nvcr.io/nvidia/pytorch:24.02-py3"
MOUNT="$OPENRLHF_PATH:/openrlhf,$HOME/.cache:/root/.cache"
GPUS_PER_NODE=8
JOBLOG="$(pwd)/logs/$training_script-$SLURM_JOB_ID.log"

readonly training_commands=" \
    openrlhf.cli.train_ppo_offline \
    --save_path ./checkpoint/gemma-2-2b-ppo_offline \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --train_batch_size 256 \
    --micro_train_batch_size 1 \
    --pretrain OpenRLHF/google-gemma-2-2b-it \
    --bf16 \
    --max_epochs 1 \
    --max_len 8192 \
    --zero_stage 3 \
    --learning_rate 9e-6 \
    --beta 0.1 \
    --prompt_data ./data/wscore.py \
    --input_key "prompt" \
    --output_key "responses" \
    --reward_key "scores" \
    --apply_chat_template \
    --max_samples 100000 \
    --reward_normalization "reward_only_rloo" \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --n_samples_per_prompt 10" \
    --use_mlflow \
    --mlflow_tracking_uri $MLFLOW_TRACKING_URI \
    --mlflow_experiment_name "ppo_offline" \
    --mlflow_run_name "ppo_offline_run_$(date +%Y%m%d_%H%M%S)"

echo $training_commands &>> ${JOBLOG}

# Job start
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# master addr and port
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

if [[ "${RUN_ENV}" = "datacenter" ]]; then
        export EXTRA_CONTAINER_MOUNT=",/ceph:/ceph"
        export CPU_BIND=""
else
        export EXTRA_CONTAINER_MOUNT=",/opt/hpcx:/opt/hpcx"
        export CPU_BIND="--cpu-bind=map_ldom:0*4,1*4"
fi

srun --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
    "cd /openrlhf; pip install . ; torchrun \
    torchrun --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT -m ${training_commands}" &>> ${JOBLOG}

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}