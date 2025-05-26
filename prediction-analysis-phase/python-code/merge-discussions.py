#!/usr/bin/env python3
import json

input_files = ["split_predictions/predictions-part_1.json",
               "split_predictions/predictions-part_2.json",
               "split_predictions/predictions-part_3.json",
               "split_predictions/predictions-part_4.json"]
combined = []
for fname in input_files:
    with open(fname, "r", encoding="utf-8") as f:
        data = json.load(f)
    combined.extend(data)
output_file = "predicted-dataset.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)
