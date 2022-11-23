import typing
import json
import jsonlines
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Dict
SPLITS = ['train', 'valid']
def main(args):
    train_data_path: Path = args.data_dir / 'train.jsonl'
    with jsonlines.open(train_data_path) as reader:
        data = [obj for obj in reader]


    train, valid = train_test_split(data, test_size=0.1, shuffle=False)
    org_data = {'train': train, 'valid': valid}
    
    data: Dict[str, List[Dict]] = {split: [{'text': new['maintext'], 'summary': new['title']} for new in org_data[split]] for split in SPLITS}

    output_paths: Dict[str, Path] = {split: args.output_dir / f'{split}.jsonl' for split in SPLITS}
    output_paths_json: Dict[str, Path] = {split: args.output_dir / f'{split}.json' for split in SPLITS}

    for split in SPLITS:
        with jsonlines.open(output_paths[split], mode='w') as writer:
            for line in data[split]:
                writer.write(line)
        output_paths_json[split].write_text(json.dumps(data[split], indent=2, ensure_ascii=False), encoding='utf-8')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data",
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
