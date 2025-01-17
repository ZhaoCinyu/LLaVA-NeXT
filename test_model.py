from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import os

os.environ["HF_HOME"] = "/playpen/xinyu"
# pretrained = '/playpen/xinyu/checkpoints/llavanext-google_siglip-so400m-patch14-384-_playpen_xinyu_checkpoints_sft_smollm_diff_v2-finetune'
# pretrained = '/playpen/xinyu/checkpoints/llavanext-google_siglip-so400m-patch14-384-_playpen_xinyu_checkpoints_sft_smollm_diff_v2-pretrain-test-official'
# pretrained = "/playpen/xinyu/checkpoints/llavanext-google_siglip-so400m-patch14-384-_playpen_xinyu_pythia_20k_pythia-70m-20k-hf-finetune-gqa"
# pretrained = "/playpen/xinyu/checkpoints/llavanext-google_siglip-so400m-patch14-384-_home_xinyuzh_unites1_pythia_20k_pythia-70m-diff-20k-hf-finetune-gqa"
pretrained = "/home/xinyuzh/unites1/checkpoints/llavanext-google_siglip-so400m-patch14-384-HuggingFaceTB_SmolLM2-135M-Instruct-finetune-clip-vision"
model_name = "llava_llama"
# pretrained = "liuhaotian--llava-v1.5-7b"
# model_name = "llava-v1.5"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device_map,
    attn_implementation='eager')
    # attn_implementation='flash_attention_2') 
    # attn_implementation='differential') # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

# conv_template = "vicuna_v1" # Make sure you use correct chat template for different models
conv_template = "smollm"
question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=256,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs)