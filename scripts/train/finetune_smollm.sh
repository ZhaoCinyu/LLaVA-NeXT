REPORT_TO=${1:-"none"}
BATCH_PROCESSOR_SIZE=${2:-"16"}

export HF_HOME='/playpen/xinyu'
export CUDA_VISIBLE_DEVICES=3,4,6,7
export DS_SKIP_CUDA_CHECK=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

# DATA_PREFIX='/playpen/xinyu'
DATA_PREFIX='/home/xinyuzh/unites1'

# LLM_VERSION="EleutherAI/pythia-70m"
# LLM_VERSION="lomahony/eleuther-pythia70m-hh-sft"
LLM_VERSION="HuggingFaceTB/SmolLM2-135M-Instruct"
# LLM_VERSION="HuggingFaceTB/SmolLM2-360M-Instruct"
# LLM_VERSION="${DATA_PREFIX}/checkpoints/sft_smollm_base_v2"

LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
# VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
if [[ $VISION_MODEL_VERSION == *"clip"* ]]; then
    LR=2e-5 
    # $BATCH_PROCESSOR_SIZE=16
else
    LR=1e-5
    # $BATCH_PROCESSOR_SIZE=4
fi
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION="smollm"

<<<<<<< HEAD
BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-pretrain-clip" #-attn-pt"
BASE_RUN_NAME_U1="llavanext-google_siglip-so400m-patch14-384-HuggingFaceTB_SmolLM2-135M-Instruct-pretrain-clip"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-finetune-ss" #-attn-pt"
=======
BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-pretrain-test-official" #-attn-pt"
# BASE_RUN_NAME_U1="llavanext-google_siglip-so400m-patch14-384-_playpen_xinyu_checkpoints_sft_smollm_base_v2-pretrain"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-pretrain-test-official" #-attn-pt"
>>>>>>> fb67d1d (finalize smollm)
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint
# CKPT_PATH='/playpen/xinyu'
<<<<<<< HEAD

=======
CKPT_PATH='/playpen/xinyu/checkpoints/projectors/llavanext-google_siglip-so400m-patch14-384-HuggingFaceTB_SmolLM2-135M-Instruct-pretrain-test-official'
>>>>>>> fb67d1d (finalize smollm)
NUM_GPUS=4
NNODES=1
PORT=29500
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $LLM_VERSION \
    --version $PROMPT_VERSION \
    --data_path ${DATA_PREFIX}/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
    --image_folder ${DATA_PREFIX}/LLaVA-Instruct-150K/images \
<<<<<<< HEAD
    --pretrain_mm_mlp_adapter ${DATA_PREFIX}/checkpoints/projectors/$BASE_RUN_NAME_U1/mm_projector.bin \
=======
    --pretrain_mm_mlp_adapter ${CKPT_PATH}/mm_projector.bin \
>>>>>>> fb67d1d (finalize smollm)
    --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /playpen/xinyu/checkpoints/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PROCESSOR_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
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
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --run_name $MID_RUN_NAME \
<<<<<<< HEAD
    --attn_implementation eager
=======
    # --attn_implementation differential
>>>>>>> fb67d1d (finalize smollm)

exit 0;
