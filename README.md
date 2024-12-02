

# LLaVa-Med SFT for X-ray Report Generation. (Autumn Project for Sii.)
## Overview
We provide a baseline for x-ray report generation based on [LLaVa-med](https://github.com/microsoft/LLaVA-Med) on the MIMIC dataset. For a more efficient way to build a stronger report generation model, we look forward to your exploration.

## Contents
- [Dataset](#Dataset)
- [Install](#install)
- [Training](#Training)
- [Inference](#inference)
- [Metrics](#metrics)

## Dataset
### LLaVa-Med Dataset
| Alignment data files | Size |
| --- | ---: |
| [llava_med_alignment_500k.json](https://hanoverprod.z21.web.core.windows.net/med_llava/alignment/llava_med_alignment_500k.json) | 341.52 MiB |

| Instruction-Tuning data files | Size |
| --- | ---: |
| [llava_med_instruct_10k.json](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_10k.json) | 19.24 MiB |
| [llava_med_instruct_60k.json](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_60k.json) | 	84.65 MiB |
| [llava_med_instruct_60k_inline_mention.json](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_60k_inline_mention.json) | 83.61 MiB |
| [llava_med_instruct_fig_captions.json](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_fig_captions.json) | 161.39 MiB |

| Evaluation files | Size |
| --- | ---: |
| [llava_med_eval_qa50_qa.jsonl](https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_eval_qa50_qa.jsonl) | 	256.18 KiB |
| [llava_med_eval_qa50_fig_captions.json](https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_eval_qa50_fig_captions.json) | 51.82 KiB |
| [llava_med_qa50_instruct_caption_in_text_cleaned-60k-3epoch.json](https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_qa50_instruct_caption_in_text_cleaned-60k-3epoch.json) | 100.97 KiB |

| Image URLS | Size |
| --- | ---: |
| [llava_med_image_urls.jsonl](https://hanoverprod.z21.web.core.windows.net/med_llava/llava_med_image_urls.jsonl) | 122.82 MiB |

[download_images.py](llava/data/download_images.py) is used to download the PMC articles using the above image_urls file and extract the images

To download langauge-image multimodal instruction-folllowing dataset, please run the following script:
```bash
sh download_data.sh
```
### MIMIC Dataset
We provide the MIMIC dataset in cluster of SII, with the following file structure:

```
data/
└── mimic_sft/
    ├── images/
    │   ├── 0001-0001.png
    │   └── 0001-0002.png
    │   └── 0002-0001.png
    └── mimic-cxr-2.0.0-metadata.csv
```
- The image file names follow the format {study_id}-{image_id}.png, for example, 0001-0001.png. Here, {study_id} denotes the study number, while {image_id} indicates the view number within a study, such as AP or PA. Lateral views are excluded, consistent with previous research, for our baseline training. However, we provide all view images, allowing you to design algorithms that fully utilize this data.
- Following [MGCA's procedure](https://github.com/HKU-MedAI/MGCA/blob/main/mgca/preprocess/mimic_cxr.py) to pre-process the MIMIC-CXR dataset.[exclude the lateral view data and do data cleaning]

## Install

1. Clone this repository and navigate to LLaVA-Med folder
```bash
https://github.com/microsoft/LLaVA-Med.git
cd LLaVA-Med
```

2. Install Package: Create conda environment

```Shell
conda create -n llava-med python=3.10 -y
conda activate llava-med
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
3. Install additional packages for training cases
- You can install packages on the web cluster of sii and set the needed python path.
```
pip uninstall torch torchvision -y
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install openai==0.27.8
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers@cae78c46
pip install -e .
pip install einops ninja open-clip-torch
pip install flash-attn --no-build-isolation
```
<!-- ## Model Download
-  [microsoft/llava-med-v1.5-mistral-7b](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) -->

## Training

### Initialization from LLaVA-7B Weights

To ensure the smooth adaptation in terms of the multimodal chat capability, we initialize model weights from the general-domain [LLaVA](https://llava-vl.github.io/). The delta weights of LLaVA comply with the LLaMA model license. You can add the delta to the original LLaMA weights to obtain the LLaVA weights.

1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).(If you can not download the target model, we'll offer the same one.)
2. Use the following scripts to get LLaVA weights ``LLaVA-7b-v0'' by applying our delta [LLaVA-7b-delta-v0](https://huggingface.co/liuhaotian/LLaVA-7b-delta-v0)). It will automatically download delta weights from our Hugging Face account.

This conversion command needs around 30 GB of CPU RAM.
```bash
python3 -m llava.model.apply_delta \
    --base /path/to/llama-7b \
    --target /output/path/to/LLaVA-7b-v0 \
    --delta /huggingface.co/liuhaotian/LLaVA-7b-delta-v0
```
For simplicity, we directly use the LLaVA-Med aligned and [instruction-tuned weights](https://hanoverprod.z21.web.core.windows.net/med_llava/models/llava_med_in_text_60k_ckpt2_delta.zip) as the initialization weight for the report generation model. Also, you need to run script above for merging delta weight.

### LLaVA-Med Training
LLaVA-Med is trained on 8 H100 GPUs with 80GB memory with the following code. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly to keep the global batch size the same.

<details>
<summary>SFT: LLaVA-Med-7B, 8x H100 (80G).  Time: ~2 hours.</summary>

```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path ./checkpoints/llava-7b-v0 \
    --data_path /path/to/pubmed_600k.json \
    --image_folder /path/to/pubmed_600k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/llava-med-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none
```
</details>

## Inference
run [bash inference.sh](inference.sh) for generating report on images with prompts.
--num-chunks: number of gpus
--model-name: path/to/your/model
--image-folder: image folder
--answers-file-pred: target save prediction csv path
--answers-file-gt: target save ground-truth csv path
--question-file 

## Metrics
Adapted from [link](https://github.com/rajpurkarlab/CXR-Report-Metric/). We make some modifications for better execution. Metrics are:
* BLEU(1,2,3,4)
* BERTscore
* CheXbert labeler vector similarity

Notes:
- You can move to their repo for detailed intructions. Attention we have make some modifications in [CXR-Report-Metric](CXR-Report-Metric) for better compability with our datasets.
- Models are provided in cluster of sii.

### Final Report
Ground Truth and Predicted reports must be arranged in the same order in a
column named "report" in two CSV files. The CSVs should also contain a
corresponding "study_id" column that contains unique identifies for the reports.

### Usage
```
cd CXR-Report-Metric
bash cal_metric.sh
```



