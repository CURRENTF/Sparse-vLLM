#!/bin/bash

# 执行一次 git pull
echo "Updating code..."
git pull

# 基础路径和固定参数
MODEL_PATH="/root/autodl-fs/models/Llama-3.1-8B-Instruct"
DATASET_PATH="/root/autodl-fs/datasets/deltakv_llama3_train_num40000_seqlen8192"
BASE_OUTPUT_DIR="/root/autodl-fs/checkpoints/compressor"
BASE_WANDB_GROUP="llama_hyperparams_ablation_v1"

# 默认（基准）值
DEFAULT_KV_SIZE=512
DEFAULT_SEQ_CHUNK=4
DEFAULT_INTER_SIZE=4096
DEFAULT_CLUSTER_RATIO=0.1

# 定义一个运行训练的函数，方便重复调用
run_train() {
    local kv_size=$1
    local seq_chunk=$2
    local inter_size=$3
    local cluster_ratio=$4
    local tag=$5
    local gradient_checkpointing=${6:-False}

    echo "----------------------------------------------------------------"
    echo "Running ablation: $tag (KV=$kv_size, Seq=$seq_chunk, Inter=$inter_size, Ratio=$cluster_ratio, GC=$gradient_checkpointing)"
    echo "----------------------------------------------------------------"

    python src/deltakv/train_compressor.py \
        --model_name_or_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_dir "${BASE_OUTPUT_DIR}" \
        --kv_compressed_size "$kv_size" \
        --seq_chunk_size "$seq_chunk" \
        --layer_chunk_size 1 \
        --batch_size 1 \
        --warmup_ratio 0.02 \
        --max_steps 5000 \
        --learning_rate 2e-4 \
        --use_nonlinear_compressor True \
        --ref_mode avg \
        --collect_kv_before_rope True \
        --model_type cluster_e2e_big \
        --cluster_soft_assignment False \
        --compressor_intermediate_size "$inter_size" \
        --cluster_ratio "$cluster_ratio" \
        --wandb_group "${BASE_WANDB_GROUP}" \
        --save_total_limit 1 \
        --gradient_checkpointing "$gradient_checkpointing"

    echo "Finished: $tag"
}

# 1. 消融 kv_compressed_size: 128, 256, 384, 512, 768
#for val in 128 256 384 512 768; do
#    run_train $val $DEFAULT_SEQ_CHUNK $DEFAULT_INTER_SIZE $DEFAULT_CLUSTER_RATIO "kv_size_$val"
#done

# 2. 消融 seq_chunk_size: 2, 4, 8, 16
#for val in 2 4 8 16; do
#    # 跳过已经跑过的基准值 (4, 已经在上面的循环或者基准里涵盖)
#    if [ $val -eq $DEFAULT_SEQ_CHUNK ]; then continue; fi
#    run_train $DEFAULT_KV_SIZE $val $DEFAULT_INTER_SIZE $DEFAULT_CLUSTER_RATIO "seq_chunk_$val"
#done

# 3. 消融 compressor_intermediate_size: 1024, 2048, 3072, 4096, 6144
#for val in 1024 2048 3072 4096 6144; do
#    # 跳过基准值 (4096)
#    if [ $val -eq $DEFAULT_INTER_SIZE ]; then continue; fi
#    run_train $DEFAULT_KV_SIZE $DEFAULT_SEQ_CHUNK $val $DEFAULT_CLUSTER_RATIO "inter_size_$val"
#done

# 4. 消融 cluster ratio
#for val in 0.05 0.1 0.2 0.3; do
#    # 跳过基准值 (已经由其它实验覆盖或者作为对比)
#     if [ $val == $DEFAULT_CLUSTER_RATIO ]; then continue; fi
#    run_train $DEFAULT_KV_SIZE $DEFAULT_SEQ_CHUNK $DEFAULT_INTER_SIZE $val "cluster_ratio_$val"
#done


# 下面是补充的一些实验
export FORCE_QWEN=1 # 走Qwen big
#val=1024
#run_train $val $DEFAULT_SEQ_CHUNK $DEFAULT_INTER_SIZE $DEFAULT_CLUSTER_RATIO "kv_size_$val"

#val=32
#run_train $DEFAULT_KV_SIZE $val $DEFAULT_INTER_SIZE $DEFAULT_CLUSTER_RATIO "seq_chunk_$val"
#
#val=1.0
#run_train $DEFAULT_KV_SIZE $DEFAULT_SEQ_CHUNK $DEFAULT_INTER_SIZE $val "cluster_ratio_$val"

# 3. 消融 compressor_intermediate_size: 1024, 2048, 3072, 4096, 6144
for val in 6144 8192; do
    # 跳过基准值 (4096)
    if [ $val -eq $DEFAULT_INTER_SIZE ]; then continue; fi
    run_train $DEFAULT_KV_SIZE $DEFAULT_SEQ_CHUNK $val $DEFAULT_CLUSTER_RATIO "inter_size_$val" True
done

echo "All ablation experiments completed!"