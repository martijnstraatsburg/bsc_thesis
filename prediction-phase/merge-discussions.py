#!/usr/bin/env python3
import json

input_files = ["preds-part1.json", "preds-part2.json"]
combined = []
for fname in input_files:
    with open(fname, "r", encoding="utf-8") as f:
        data = json.load(f)
    combined.extend(data)
output_file = "1000-discussions-predictions.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)
