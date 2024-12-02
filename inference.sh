export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/myconda/conda_dir/envs/local_env/bin:$PATH"

# python llava/eval/model_vqa_mimic.py \
#     --model-name checkpoints/llava_mimic_sft_nolateral_bs128_ep3 \
#     --image-folder data/MIMIC-CXR/images_processed/images \
#     --answers-file-pred data/answer/llava_mimic_sft_nolateral_bs128_ep3/pred.csv \
#     --answers-file-gt data/answer/llava_mimic_sft_nolateral_bs128_ep3/gt.csv \
#     --question-file data/mimic_sft/valid.json \

#         multi-gpu   
# python llava/eval/run_med_datasets_eval_batch.py \
#     --num-chunks 8 \
#     --model-name checkpoints/llava_mimic_sft_findings_nolateral_bs128_ep3 \
#     --image-folder data/MIMIC-CXR/images_processed/images \
#     --answers-file-pred data/answer/llava_mimic_sft_findings_nolateral_bs128_ep3/pred.csv \
#     --answers-file-gt data/answer/llava_mimic_sft_findings_nolateral_bs128_ep3/gt.csv \
#     --question-file data/mimic_sft/valid_findings.json

python llava/eval/run_med_datasets_eval_batch.py \
    --num-chunks 8 \
    --model-name checkpoints/llava_mimic_sft_impression_nolateral_bs128_ep3 \
    --image-folder data/MIMIC-CXR/images_processed/images \
    --answers-file-pred data/answer/llava_mimic_sft_impression_nolateral_bs128_ep3/pred.csv \
    --answers-file-gt data/answer/llava_mimic_sft_impression_nolateral_bs128_ep3/gt.csv \
    --question-file data/mimic_sft/valid_impression.json
