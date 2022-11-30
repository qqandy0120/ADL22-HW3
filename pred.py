import argparse
import wandb
import json
from tqdm import tqdm
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
)
import os
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    
    # load data
    data = load_dataset('json', data_files={'eval': args.eval_path})
    dataset = data['eval']

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # create model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(DEVICE)

    def postprocess_text(preds):
        ch_mapping = {
            ',': '，',
            '!': '！',
            ':': '：',
            '?': '？', 
            '/': '／',
            '%': '％',
        }
        for ch in ch_mapping.keys():
            if ch in preds:
                preds = preds.replace(ch, ch_mapping[ch])
                
        return preds

    padding = 'max_length'
    with open(args.output_path, 'w') as f:
        for i, batch in enumerate(tqdm(dataset)):
            inputs = tokenizer(batch['maintext'], max_length=args.max_source_length, padding=padding, return_tensors='pt', truncation=True).to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=args.max_target_length,num_beams=5,repetition_penalty=2.5)
            decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            decoded_pred = postprocess_text(decoded_pred)
            print(decoded_pred)

            json.dump({
                'title': decoded_pred,
                'id': batch['id']},
                f,
                ensure_ascii=False
            )
            f.write('\n')
            if i == 10:
                break

    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_path",
        default="./data/public.jsonl",
        type=str,
    )
    parser.add_argument(
        "--model_name_or_path",
        default="checkpoint-25000",
        type=str,
    )
    parser.add_argument(
        "--max_source_length",
        default=512,
        type=int,
    )
    parser.add_argument(
        "--max_target_length",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to output file"
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)