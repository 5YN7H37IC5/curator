#!/usr/bin/env python3
"""
Generate a Q&A dataset for OpenAI fine-tuning.

This script uses the hierarchical generators defined in ungrounded_qa.py
to generate a specified number of question-answer pairs, then splits
the data into training and validation sets and saves them as JSONL files.
"""

import os
import sys
import argparse
import random
import json

# Ensure the ungrounded_qa module is on the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from ungrounded_qa import SubjectGenerator, SubsubjectGenerator, QAGenerator


def generate_qa_pairs(total_count, model_name):
    """Generate a list of prompt-completion dicts up to total_count."""
    results = []
    while len(results) < total_count:
        # Generate subjects
        subjects = SubjectGenerator(model_name=model_name)()
        # Generate subsubjects
        subsubjects = SubsubjectGenerator(model_name=model_name)(subjects)
        # Generate Q&A pairs
        qa_dataset = QAGenerator(model_name=model_name)(subsubjects)
        # Iterate rows from the HuggingFace Dataset
        for row in qa_dataset:
            prompt = f"{row['question']}\nA:"
            completion = f" {row['answer']}"
            results.append({'prompt': prompt, 'completion': completion})
            if len(results) >= total_count:
                break
    return results[:total_count]


def split_and_save(data, train_count, val_count, output_dir, seed=42):
    """Shuffle, split, and save data into train/validation JSONL files."""
    random.Random(seed).shuffle(data)
    assert len(data) >= train_count + val_count, (
        f"Need at least {train_count+val_count} samples, got {len(data)}"
    )
    train = data[:train_count]
    valid = data[train_count:train_count+val_count]

    train_path = os.path.join(output_dir, 'train.jsonl')
    valid_path = os.path.join(output_dir, 'valid.jsonl')

    with open(train_path, 'w') as ft:
        for item in train:
            ft.write(json.dumps(item) + '\n')

    with open(valid_path, 'w') as fv:
        for item in valid:
            fv.write(json.dumps(item) + '\n')

    print(f"Saved {len(train)} samples to {train_path}")
    print(f"Saved {len(valid)} samples to {valid_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate and split a Q&A dataset for fine-tuning.')
    parser.add_argument('--train-samples', type=int, default=1111,
                        help='Number of training samples.')
    parser.add_argument('--val-samples', type=int, default=1111,
                        help='Number of validation samples.')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model name to use for generation.')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save the JSONL files.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    total = args.train_samples + args.val_samples
    print(f"Generating {total} Q&A pairs with model {args.model}...")
    data = generate_qa_pairs(total, args.model)
    split_and_save(data, args.train_samples, args.val_samples,
                   args.output_dir, seed=args.seed)


if __name__ == '__main__':
    main()