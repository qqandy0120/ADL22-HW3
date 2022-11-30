# ADL22-HW3

## download dataset and trained model
```bash
bash download.sh
```

## train data on mt5-small
```bash
python train.py --train_path path/to/train.jsonl --eval_path path/to/public.jsonl
```

## do prediction
```bash
run.sh {/path/to/input.jsonl} {/path/to/output.jsonl}
```