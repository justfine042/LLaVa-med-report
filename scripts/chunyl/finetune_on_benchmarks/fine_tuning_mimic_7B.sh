export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=":$PATH"
# python \
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path composed_weights/llava_med_in_text_60k_ckpt2_delta \
    --data_path data/mimic_sft/train.json \
    --image_folder data/MIMIC-CXR/images_processed/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir ./checkpoints/mimic_sft_nolateral_bs128_ep3 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none    

    # --pretrain_mm_mlp_adapter /home/chunyl/research/models/llava/LLaVA-13b-pretrain-projector-v0/LLaVA-13b-pretrain-projector-v0-CC3M-595K-original_caption.bin \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
