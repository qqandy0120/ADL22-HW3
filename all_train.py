import typing
import json
import jsonlines
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Dict


# create test data
test_data_path: Path = Path('data') / 'train.jsonl'
with jsonlines.open(test_data_path) as reader:
    test_data = [obj for obj in reader]
test_data = [{'text': new['maintext'], 'summary': new['title']} for new in test_data]
test_output_path = Path('cache') / 'alltrain.jsonl'
test_output_path_json =  Path('cache') / 'alltrain.json'
with jsonlines.open(test_output_path, mode='w') as writer:
    for line in test_data:
            writer.write(line)
test_output_path_json.write_text(json.dumps(test_data, indent=2, ensure_ascii=False), encoding='utf-8')