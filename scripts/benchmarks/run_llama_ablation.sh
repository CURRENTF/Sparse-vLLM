#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_ROOT="${DELTAKV_MODEL_ROOT:-${REPO_ROOT}/models}"
DATA_ROOT="${DELTAKV_DATA_DIR:-${REPO_ROOT}/data}"
COMPRESSOR_ROOT="${DELTAKV_COMPRESSOR_ROOT:-${REPO_ROOT}/checkpoints/compressor}"

# 执行一次 git pull
echo "Updating code..."
git -C "${REPO_ROOT}" pull --ff-only

# 基础路径和固定参数
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Llama-3.1-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/deltakv_llama3_train_num40000_seqlen8192}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${COMPRESSOR_ROOT}}"
BASE_WANDB_GROUP="llama_hyperparams_ablation_v1"

# 默认（基准）值
DEFAULT_LATENT_DIM=512
DEFAULT_NEIGHBOR_COUNT=4
DEFAULT_INTER_SIZE=4096
DEFAULT_CENTER_RATIO=0.1

# 定义一个运行训练的函数，方便重复调用
run_train() {
    local latent_dim=$1
    local neighbor_count=$2
    local inter_size=$3
    local center_ratio=$4
    local tag=$5
    local gradient_checkpointing=${6:-False}

    echo "----------------------------------------------------------------"
    echo "Running ablation: $tag (latent_dim=$latent_dim, neighbor_count=$neighbor_count, Inter=$inter_size, Ratio=$center_ratio, GC=$gradient_checkpointing)"
    echo "----------------------------------------------------------------"

    python src/deltakv/train_compressor.py \
        --model_name_or_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_dir "${BASE_OUTPUT_DIR}" \
        --deltakv_latent_dim "$latent_dim" \
        --deltakv_neighbor_count "$neighbor_count" \
        --layer_chunk_size 1 \
        --batch_size 1 \
        --warmup_ratio 0.02 \
        --max_steps 5000 \
        --learning_rate 2e-4 \
        --use_nonlinear_compressor True \
        --collect_kv_before_rope True \
        --model_type cluster_e2e_big \
        --cluster_soft_assignment False \
        --compressor_intermediate_size "$inter_size" \
        --deltakv_center_ratio "$center_ratio" \
        --wandb_group "${BASE_WANDB_GROUP}" \
        --save_total_limit 1 \
        --gradient_checkpointing "$gradient_checkpointing"

    echo "Finished: $tag"
}

# 1. 消融 deltakv_latent_dim: 128, 256, 384, 512, 768
#for val in 128 256 384 512 768; do
#    run_train $val $DEFAULT_NEIGHBOR_COUNT $DEFAULT_INTER_SIZE $DEFAULT_CENTER_RATIO "latent_dim_$val"
#done

# 2. 消融 deltakv_neighbor_count: 2, 4, 8, 16
#for val in 2 4 8 16; do
#    # 跳过已经跑过的基准值 (4, 已经在上面的循环或者基准里涵盖)
#    if [ $val -eq $DEFAULT_NEIGHBOR_COUNT ]; then continue; fi
#    run_train $DEFAULT_LATENT_DIM $val $DEFAULT_INTER_SIZE $DEFAULT_CENTER_RATIO "neighbor_count_$val"
#done

# 3. 消融 compressor_intermediate_size: 1024, 2048, 3072, 4096, 6144
#for val in 1024 2048 3072 4096 6144; do
#    # 跳过基准值 (4096)
#    if [ $val -eq $DEFAULT_INTER_SIZE ]; then continue; fi
#    run_train $DEFAULT_LATENT_DIM $DEFAULT_NEIGHBOR_COUNT $val $DEFAULT_CENTER_RATIO "inter_size_$val"
#done

# 4. 消融 deltakv_center_ratio
#for val in 0.05 0.1 0.2 0.3; do
#    # 跳过基准值 (已经由其它实验覆盖或者作为对比)
#     if [ $val == $DEFAULT_CENTER_RATIO ]; then continue; fi
#    run_train $DEFAULT_LATENT_DIM $DEFAULT_NEIGHBOR_COUNT $DEFAULT_INTER_SIZE $val "center_ratio_$val"
#done


# 下面是补充的一些实验
export FORCE_QWEN=1 # 走Qwen big
#val=1024
#run_train $val $DEFAULT_NEIGHBOR_COUNT $DEFAULT_INTER_SIZE $DEFAULT_CENTER_RATIO "latent_dim_$val"

#val=32
#run_train $DEFAULT_LATENT_DIM $val $DEFAULT_INTER_SIZE $DEFAULT_CENTER_RATIO "neighbor_count_$val"
#
#val=1.0
#run_train $DEFAULT_LATENT_DIM $DEFAULT_NEIGHBOR_COUNT $DEFAULT_INTER_SIZE $val "center_ratio_$val"

# 3. 消融 compressor_intermediate_size: 1024, 2048, 3072, 4096, 6144
for val in 6144 8192; do
    # 跳过基准值 (4096)
    if [ $val -eq $DEFAULT_INTER_SIZE ]; then continue; fi
    run_train $DEFAULT_LATENT_DIM $DEFAULT_NEIGHBOR_COUNT $val $DEFAULT_CENTER_RATIO "inter_size_$val" True
done

echo "All ablation experiments completed!"
