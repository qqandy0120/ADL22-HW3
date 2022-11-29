import argparse
import wandb

from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    set_seed,
)

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# os.environ['TRANSFORMERS_CACHE'] = '/tmp2/b08902038/cache'
# os.environ['HF_DATASETS_CACHE'] = '/tmp2/b08902038/cache'

from tw_rouge import get_rouge

def main():
    args = parse_args()
    raw_data = load_dataset('json', data_files={'train': args.train_path, 'eval': args.eval_path})
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess(example):
        maintext = example['maintext']
        summary = example['title']
        inputs = tokenizer(maintext, max_length=args.max_input_length, padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(summary, max_length=args.max_target_length, padding="max_length", truncation=True)
        inputs['labels'] = labels['input_ids']
        return inputs
    
    train_dataset = raw_data['train'].map(preprocess, batched=True, remove_columns=['date_publish', 'maintext', 'source_domain', 'split', 'title', 'id'])
    eval_dataset = raw_data['eval'].map(preprocess, batched=True, remove_columns=['date_publish', 'maintext', 'source_domain', 'split', 'title', 'id'])
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = [pred.replace(':', '：').replace('?', '？').replace('!', '！').replace(',', '，').replace('/', '／').strip() + '\n' for pred in decoded_preds]
        labels = []
        import json
        with open(args.eval_path) as file:
            for line in file:
                line = json.loads(line)
                labels.append(line['title'].strip() + '\n')

        result = get_rouge(decoded_preds, labels)
        result = {key: round(value['f']*100, 6) for key, value in result.items()}
        return result

    trainer_args = Seq2SeqTrainingArguments(
        output_dir="cache/how",
        save_strategy = "steps",
        save_steps=1000,
        evaluation_strategy='steps',
        eval_steps=500,
        report_to="wandb",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        seed=309713,
        max_steps=15000,
        predict_with_generate=True,
        generation_max_length=128,
        gradient_accumulation_steps=args.gradient_accumulation,
        logging_strategy='steps',
        logging_steps=50,
    )
    trainer = Seq2SeqTrainer(
        model,
        trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    # wandb.init(project="huggingface", id="jxt9y0lo", resume="must")
    if (args.model != 'google/mt5-small'):
        trainer.train(args.model)
    else:
        trainer.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        default="data/train.jsonl",
        type=str,
        help="path to train data"
    )
    parser.add_argument(
        "--eval_path",
        default="data/public.jsonl",
        type=str,
        help="path to train data"
    )
    parser.add_argument(
        "--model",
        default="google/mt5-small",
        type=str,
        help="model type",
    )
    parser.add_argument(
        "--max_input_length",
        default=256,
        type=int,
        help="max length of input"
    )
    parser.add_argument(
        "--max_target_length",
        default=64,
        type=int,
        help="max length of target"
    )
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="batch size"
    )
    parser.add_argument(
        "--gradient_accumulation",
        default=16,
        type=int,
        help="gradient accumulation"
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("yes")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    set_seed(309713)
    main()
    