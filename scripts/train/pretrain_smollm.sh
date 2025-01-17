REPORT_TO=${1:-"none"}
BATCH_PROCESSOR_SIZE=${2:-"16"}
DATA_PREFIX='/home/xinyuzh/unites1'
# DATA_PREFIX='/playpen/xinyu'

export HF_HOME='/playpen/xinyu'
export CUDA_VISIBLE_DEVICES=1,2,3,4

# LLM_VERSION="EleutherAI/pythia-70m"
# LLM_VERSION="lomahony/eleuther-pythia70m-hh-sft"
LLM_VERSION="${DATA_PREFIX}/checkpoints/sft_smollm_base_v2"
# LLM_VERSION="HuggingFaceTB/SmolLM2-360M-Instruct"
# LLM_VERSION="HuggingFaceTB/SmolLM2-1.7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
# VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
if [[ $VISION_MODEL_VERSION == *"clip"* ]]; then
    LR=2e-3 
else
    LR=1e-3
fi
 # for siglip
# LR=2e-3 # for clip
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-pretrain-clip"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

NUM_GPUS=4
NNODES=1
PORT=29500

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --verbose_logging \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PREFIX}/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ${DATA_PREFIX}/LLaVA-Pretrain/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /playpen/xinyu/checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PROCESSOR_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers $BATCH_PROCESSOR_SIZE \
    --lazy_preprocess True \
    --report_to $REPORT_TO \
    --run_name $BASE_RUN_NAME

# You can delete the sdpa attn_implementation if you want to use flash attn