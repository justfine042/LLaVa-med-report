export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/myconda/conda_dir/envs/llava-med-report/bin:$PATH"
python llava/eval/model_vqa_reports.py \
    --num-chunks 1 \
    --model-name checkpoints/llava_med_in_text_reprot_gen_bs1_tune_proj_gpu2 \
    --image-folder 'private/images' \
    --answers-file-pred ./pred_private_bs1.csv \
    --answers-file-gt ./gt_private_bs1.csv \
    --question-file xray-report/private_test.json \
    --batch-size 64


   

    
