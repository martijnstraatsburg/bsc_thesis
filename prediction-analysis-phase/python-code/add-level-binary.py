#!/usr/bin/env python3

# Author: Martijn Straatsburg
# Name: add-level-binary.py
# Description: ...

import json
import sys
import os

INPUT_FILE = 'predicted-dataset.json'
OUTPUT_FILE = 'predicted-dataset-updated.json'

def map_level(value):
    if value in (1, 2):
        return 'low'
    elif value == 3:
        return 'medium'
    elif value in (4, 5):
        return 'high'
    else:
        return None


def map_binary(value):
    try:
        return 'under' if value < 2.5 else 'over'
    except TypeError:
        return None


def main():
    if not os.path.isfile(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        entry['level_suspense'] = map_level(entry.get('suspense'))
        entry['level_curiosity'] = map_level(entry.get('curiosity'))
        entry['level_surprise'] = map_level(entry.get('surprise'))

        entry['binary_suspense'] = map_binary(entry.get('suspense'))
        entry['binary_curiosity'] = map_binary(entry.get('curiosity'))
        entry['binary_surprise'] = map_binary(entry.get('surprise'))

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Augmented dataset written to '{OUTPUT_FILE}'.")

if __name__ == '__main__':
    main()
