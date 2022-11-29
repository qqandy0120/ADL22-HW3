import argparse
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from tw_rouge import get_rouge
SPLITS = ['train', 'eval']

def main(args):
    # clean cache
    torch.cuda.empty_cache()

    # read file
    paths = {'train': args.train_file, 'eval': args.eval_file}
    datas = load_dataset('json', data_files={split: paths[split] for split in SPLITS})

    # create model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        prefix = ""
        padding = False
        max_target_length = args.max_target_length

        inputs = [maintext for maintext in examples['text']]
        targets = [summary for summary in examples['summary']]

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def postprocess_text(preds, labels):
        ch_mapping = {
            ',': '，',
            '!': '！',
            ':': '：',
            '?': '？', 
            '/': '／',
            '%': '％',
        }
        new_preds, new_labels = [], []
        for pred in preds:
            pred = pred.strip()
            for ch in ch_mapping.keys():
                if ch in pred:
                    pred = pred.replace(ch, ch_mapping[ch])
            new_preds.append(pred+'\n')
        for label in labels:
            label = label.strip()
            for ch in ch_mapping.keys():
                if ch in label:
                    label = label.replace(ch, ch_mapping[ch])
            new_labels.append(label+'\n')

        return new_preds, new_labels

    # customize coupute_metrics
    def customized_compute_metrics(eval_preds):
        preds, labels = eval_preds
        # print("----------------------------")
        # print(f"preds: {preds}")
        # print(f"labels: {labels}")
        # print("----------------------------")
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # ignore pad token for loss
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        # print("----------------------------")
        # print(f"decoded_pred: {decoded_preds}")
        # print(f"decode labels: {decoded_labels}")
        # print("----------------------------")
        result = get_rouge(decoded_preds, decoded_labels)
        result = {k: round(v['f'] * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    
    # create dataset
    datasets = {split: datas[split].map(
        preprocess_function,
        batched=True,
        desc=f"Running tokenizer on {split} dataset",
    ) for split in SPLITS}

    # create data collactor
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    # create trainer arguments
    trainer_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        save_strategy = "steps",
        evaluation_strategy='steps',
        eval_steps=500,
        report_to=args.report_to,
        learning_rate=4e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        seed=args.seed,
        predict_with_generate=True,
        generation_max_length=128,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_strategy='steps',
        logging_steps=50,
        num_train_epochs=args.num_train_epochs,
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=trainer_args,
        train_dataset=datasets['train'] if args.do_train else None,
        eval_dataset=datasets['eval'] if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=customized_compute_metrics,
    )

    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed',
        type=int,
        default=7777777,
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=True,
    )
    parser.add_argument(
        "--do_train",
        default=False,
        type=bool,
        help="do train on train dataset"
    )
    parser.add_argument(
        "--do_eval",
        default=False,
        type=bool,
        help="do evaluation on eval dataset"
    )
    parser.add_argument(
        "--train_file",
        default="./cache/train.json",
        type=str,
        help="path to train data"
    )
    parser.add_argument(
        "--eval_file",
        default="./cache/eval.json",
        type=str,
        help="path to eval data"
    )
    parser.add_argument(
        "--output_dir",
        default=False,
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--report_to",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--max_source_length",
        default=512,
        type=int,
    )
    parser.add_argument(
        "--max_target_length",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--fp16",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=16,
        type=int,
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)