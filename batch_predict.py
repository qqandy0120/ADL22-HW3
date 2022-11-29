import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed, DataCollatorForSeq2Seq
from tqdm import tqdm
import torch
from pathlib import Path

def main():
    args = parse_args()
    raw_data = load_dataset('json', data_files={'eval':args.eval_path})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    def preprocess(example):
        maintext = example['maintext']
        summary = example['title']
        inputs = tokenizer(maintext, max_length=args.max_input_length, padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(summary, max_length=args.max_target_length, padding="max_length", truncation=True)
        inputs['labels'] = labels['input_ids']
        return inputs
    
    eval_dataset = raw_data['eval'].map(preprocess, batched=True, remove_columns=['date_publish', 'maintext', 'source_domain', 'split', 'title', 'id'])
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)

    import json
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_dir = args.output_dir
    if (out_dir[-1] == '/'):
        out_dir = out_dir[:-1]
    fs = open(f'{args.output_dir}/{args.strategy}.jsonl', 'w+')
    gen_kwargs = {
        'greedy': {
            "min_length": 15,
            "max_length": args.max_target_length,
        },
        'beam-1': {
            "min_length": 15,
            "max_length": args.max_target_length,
            "num_beams": 5,
            "early_stopping": True,
            "repetition_penalty": 2.5,
        },
        'beam-2': {
            "min_length": 15,
            "max_length": args.max_target_length,
            "num_beams": 10,
            "early_stopping": True,
            "repetition_penalty": 2.5,
        },
        'beam-3': {
            "min_length": 15,
            "max_length": args.max_target_length,
            "num_beams": 15,
            "early_stopping": True,
            "repetition_penalty": 2.5,
        },
        'topk-1': {
            "min_length": 15,
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_k": 10,
            "repetition_penalty": 2.5,
        },
        'topk-2': {
            "min_length": 15,
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_k": 50,
            "repetition_penalty": 2.5,
        },
        'topp-1': {
            "min_length": 15,
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_k": 0,
            "top_p": 0.8,
            "repetition_penalty": 2.5,
        },
        'topp-2': {
            "min_length": 15,
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_k": 0,
            "top_p": 0.9,
            "repetition_penalty": 2.5,
        },
        'temp-1': {
            "min_length": 15,
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_k": 0,
            "temperature": 0.7,
            "repetition_penalty": 2.5,
        },
        'temp-2': {
            "min_length": 15,
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_k": 0,
            "temperature": 0.5,
            "repetition_penalty": 2.5,
        },
    }
    model.to(device)
    model.eval()

    id = 0
    for batch in tqdm(eval_dataloader):
        outputs = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            **gen_kwargs[args.strategy],
        )
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for pred in decoded_output:
            json.dump({"title": pred.replace(':', '：').replace('?', '？').replace('!', '！').replace(',', '，').replace('/', '／'), "id": f'{21710+id}'}, fs, ensure_ascii=False)
            id += 1
            fs.write('\n')
    fs.close()
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_path",
        default="/tmp2/b08902038/HW3/data/public.jsonl",
        type=str,
        help="path to eval data"
    )
    parser.add_argument(
        "--model",
        required=True,
        default="model",
        type=str,
        help="model type",
    )
    parser.add_argument(
        "--max_input_length",
        default=1024,
        type=int,
        help="max length of input"
    )
    parser.add_argument(
        "--max_target_length",
        default=128,
        type=int,
        help="max length of target"
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        help="eval batch size"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="dir to output file"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        type=str,
        help="generate strategy to use"
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
    