import typing
import json
import jsonlines
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Dict
SPLITS = ['train', 'eval']
def main(args):

    data_paths = {split: getattr(args, f'{split}_path') for split in SPLITS}
    org_datas = {}
    for split in SPLITS:
        with jsonlines.open(data_paths[split]) as reader:
            org_datas[split] = [obj for obj in reader]
    
    datas = {split: [{'text': new["maintext"], 'summary': new['title']} for new in org_datas[split]] for split in SPLITS}


    output_paths: Dict[str, Path] = {split: args.output_dir / f'{split}.jsonl' for split in SPLITS}
    output_paths_json: Dict[str, Path] = {split: args.output_dir / f'{split}.json' for split in SPLITS}

    for split in SPLITS:
        with jsonlines.open(output_paths[split], mode='w') as writer:
            for line in datas[split]:
                writer.write(line)
        output_paths_json[split].write_text(json.dumps(datas[split], indent=2, ensure_ascii=False), encoding='utf-8')



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_path",
        type=Path,
        help="Directory to the dataset.",
        default="./data/train.jsonl",
    )
    parser.add_argument(
        "--eval_path",
        type=Path,
        help="Directory to the dataset.",
        default="./data/public.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache",
    )
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
