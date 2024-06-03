#!/usr/bin/env bash
export WANDB_PROJECT=loralibi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
#python run_qa.py \
torchrun --nproc_per_node 6 run_qa.py \
    --dataset_name squad \
    --max_seq_length 512 \
    --model_name_or_path "./squadbert-base-qa-jina" \
    --trust_remote_code True \
    --jina True \
    --do_eval \
    --do_train \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --output_dir squadbert-lora \
    --auto_find_batch_size True \
    --lora True \
    --report_to wandb \
    --logging_steps 20 \
    --eval_steps 100 \
    --save_steps 500 \
    --num_train_epochs 2 \
    --evaluation_strategy steps \
    --ddp_find_unused_parameters False \
    --overwrite_output_dir True \
    --learning_rate 3e-5 \
    --position_embedding alibi \
    --max_eval_samples 4000