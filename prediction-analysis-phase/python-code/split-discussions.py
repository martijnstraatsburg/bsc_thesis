#!usr/bin/env python3

# Name: split-discussions.py
# Author: Martijn Straatsburg
# Description: This script splits the large JSON discussion file into any number of smaller parts.

import json
import os
from math import ceil


def split_json_file(input_file, output_dir, num_parts):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    total_items = len(data)
    items_per_part = ceil(total_items / num_parts)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_parts):
        start = i * items_per_part
        end = start + items_per_part
        part_data = data[start:end]
        output_path = os.path.join(output_dir, f"part_{i+1}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(part_data, f, indent=2)
        print(f"Wrote {len(part_data)} items to {output_path}")


if __name__ == "__main__":
    input_file = "prediction-analysis-phase/1000-unseen-discussions.json"
    output_dir = "split_parts"
    num_parts = 4
    # Split in two parts because it failed trying it fully in one on Habrok
    split_json_file(input_file, output_dir, num_parts)
