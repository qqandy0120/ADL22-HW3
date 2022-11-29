python summarization/run_summarization.py \
    --seed 7777777 \
    --model_name_or_path cache/tst-summarization/checkpoint-21000 \
    --do_predict \
    --validation_file ./cache/test.json \
    --source_prefix "summarize: " \
    --output_dir ./cache/test1 \
    --overwrite_output_dir \
    --num_train_epochs 35 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --predict_with_generate=True \
    --max_source_length=512 \
    --max_target_length=128 \
    --gradient_accumulation_steps 16 \
    --fp16=True \
    --optim adafactor \
    --report_to="wandb" \