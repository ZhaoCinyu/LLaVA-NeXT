# mkdir /playpen/xinyu/LLaVA-Pretrain && cd /playpen/xinyu/LLaVA-Pretrain

# wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json
# wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k_meta.json 
# wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip && unzip images.zip -d images

# mkdir /playpen/xinyu/LLaVA-Instruct-150K && 
cd /playpen/xinyu/LLaVA-Instruct-150K

# wget http://images.cocodataset.org/zips/train2017.zip && mkdir -p images/coco && unzip train2017.zip -d images/coco
# wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip && mkdir -p images/gqa && unzip images.zip -d images/gqa
wget https://huggingface.co/datasets/qnguyen3/ocr_vqa/resolve/main/ocr_vqa.zip && mkdir -p images/ocr_vqa && unzip ocr_vqa.zip -d images/ocr_vqa
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip && mkdir -p images/textvqa && unzip train_val_images.zip -d images/textvqa
mkdir -p images/vg && cd images/vg
# wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip && unzip images.zip -d VG_100K
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip && unzip images2.zip -d VG_100K_2