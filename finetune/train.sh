export CUBLAS_WORKSPACE_CONFIG=:4096:8

export CUDA_VISIBLE_DEVICES=0
# export CUDA_DEVICE_MAX_CONNECTIONS=1

set -ex

LR=1e-4
NUM_GPUS=1
TRAIN_PATH=your_data_path
OUTPUT_DIR=output
BASE_MODEL_PATH=your_model_path

mkdir -p $OUTPUT_DIR

entrypoint="finetune.py"
args="       --model_name_or_path $BASE_MODEL_PATH \
             --train_data $TRAIN_PATH \
             --output_dir $OUTPUT_DIR \
             --lora_rank 32 \
             --lora_alpha 64 \
             --lora_dropout 0.1 \
             --max_seq_length 384 \
             --preprocessing_num_workers 1 \
             --per_device_train_batch_size 4 \
             --gradient_accumulation_steps 4 \
             --warmup_ratio 0.02 \
             --weight_decay 0.01 \
             --adam_beta1 0.9 \
             --adam_beta2 0.95 \
             --adam_epsilon 1e-8 \
             --max_grad_norm 0.3 \
             --learning_rate $LR \
             --lr_scheduler_type cosine \
             --num_train_epochs 6 \
             --logging_steps 10 \
             --evaluation_strategy "no" \
             --save_strategy "steps" \
             --save_steps 250 \
             --report_to "none" \
             --seed 1234 \
             --bf16 "

log_file=${OUTPUT_DIR}/train.log
rm -f "$log_file"

set -o pipefail  # 设置这个选项以确保管道中的任何命令失败都会被捕获

#单机
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --nnode=1 ${entrypoint} ${args} "$@" 2>&1 | tee -a "$log_file"

#多机
# python -m torch.distributed.launch ${entrypoint} ${args} "$@" 2>&1 | tee -a "$log_file"

ls -lh $OUTPUT_DIR