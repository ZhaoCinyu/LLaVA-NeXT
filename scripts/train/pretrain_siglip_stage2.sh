export HF_HOME='/playpen/xinyu'
export CUDA_VISIBLE_DEVICES=4,5,6,7
LLM_VERSION="Qwen/Qwen2-0.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-pretrain-diff"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-pretrain-diff-stage2"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

NUM_GPUS=4
NNODES=1
PORT=12345
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /home/xinyuzh/unites1/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /home/xinyuzh/unites1/LLaVA-Pretrain/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --pretrain_mm_mlp_adapter="/home/xinyuzh/unites1/checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_tunable_parts="diff_attention" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /playpen/xinyu/checkpoints/${MID_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME

# You can delete the sdpa attn_implementation if you want to use flash attn