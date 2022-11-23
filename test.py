import jsonlines
with jsonlines.open('data/train.jsonl') as reader:
    for obj in reader:
        ...